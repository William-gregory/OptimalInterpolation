# Sea Ice Freeboard class - for interpolating using Gaussian Process Regression
import re
import gpflow
import numpy as np
import scipy
import warnings

import tensorflow as tf
import tensorflow_probability as tfp

from OptimalInterpolation import get_data_path
from OptimalInterpolation.data_dict import DataDict, match, to_array
from OptimalInterpolation.data_loader import DataLoader
from OptimalInterpolation.utils import WGS84toEASE2_New, \
    EASE2toWGS84_New, SMLII_mod, SGPkernel, GPR


class PurePythonGPR():
    """Pure Python GPR class - used to hold model details from pure python implementation"""

    def __init__(self,
                 x,
                 y,
                 length_scales=1.0,
                 kernel_var=1.0,
                 likeli_var=1.0,
                 kernel="Matern32"):
        assert kernel == "Matern32", "only 'Matern32' kernel handled"

        # TODO: check values, make sure hyper parameters can be concatenated together

        # just store a values as attributes
        self.x = x
        self.y = y
        self.length_scales = length_scales
        self.kernel_var = kernel_var
        self.likeli_var = likeli_var

    def optimise(self, opt_method='CG', jac=True):
        kv = np.array([self.kernel_var]) if isinstance(self.kernel_var, (float, int)) else self.kernel_var
        lv = np.array([self.likeli_var]) if isinstance(self.likeli_var, (float, int)) else self.likeli_var

        x0 = np.concatenate([self.length_scales, kv, lv])
        # take the log of x0 because the first step in SMLII is to take exp
        x0 = np.log(x0)
        res = scipy.optimize.minimize(self.SMLII,
                                      x0=x0,
                                      args=(self.x, self.y[:, 0], False, None, jac),
                                      method=opt_method,
                                      jac=jac)

        #
        pp_params = np.exp(res.x)

        self.length_scales = pp_params[:len(self.length_scales)]
        self.kernel_var = pp_params[-2]
        self.likeli_var = pp_params[-1]

        # return {"sucsess": res['success'], "marginal_loglikelihood_from_opt": res["fun"]}
        return res['success']

    def get_loglikelihood(self):
        kv = np.array([self.kernel_var]) if isinstance(self.kernel_var, (float, int)) else self.kernel_var
        lv = np.array([self.likeli_var]) if isinstance(self.likeli_var, (float, int)) else self.likeli_var

        kv = kv.reshape(1) if len(kv.shape) == 0 else kv
        lv = lv.reshape(1) if len(lv.shape) == 0 else lv

        hypers = np.concatenate([self.length_scales, kv, lv])

        # SMLII returns negative marginal log likelihood (when grad=False)
        return -self.SMLII(hypers=np.log(hypers), x=self.x, y=self.y[:, 0], approx=False, M=None, grad=False)

    def SGPkernel(self, **kwargs):
        return SGPkernel(**kwargs)

    def SMLII(self, hypers, x, y, approx=False, M=None, grad=True):
        return SMLII_mod(hypers=hypers, x=x, y=y, approx=approx, M=M, grad=grad)

    def predict(self, xs, mean=0):

        ell = self.length_scales
        sf2 = self.kernel_var
        sn2 = self.likeli_var

        res = GPR(x=self.x,
                  y=self.y,
                  xs=xs,
                  ell=ell,
                  sf2=sf2,
                  sn2=sn2,
                  mean=mean,
                  approx=False,
                  M=None,
                  returnprior=True)

        # TODO: need to confirm these
        # TODO: confirm res[0], etc, can be vectors
        # TODO: allow for mean to be vector

        out = {
            "f*": res[0].flatten(),
            "f*_var": res[1]**2,
            "y": res[0].flatten(),
            "y_var": res[1]**2 + self.likeli_var
        }
        return out



class SeaIceFreeboard(DataLoader):

    def __init__(self, grid_res="25km", sat_list=None, verbose=True,
                 length_scale_name = None):

        super().__init__(grid_res=grid_res,
                         sat_list=sat_list,
                         verbose=verbose)

        #
        assert isinstance(length_scale_name, (type(None), list, tuple, np.ndarray)), \
            f"length_scale_name needs to be None, list, tuple or ndarray"
        self.length_scale_name = np.arange(1000).tolist() if length_scale_name is None else length_scale_name

        self.parameters = None
        self.mean = None
        self.inputs = None
        self.outputs = None
        self.scale_outputs = None
        self.scale_inputs = None
        self.model = None
        self.X_tree = None
        self.X_tree_data = None
        self.input_data_all = None
        self.valid_gpr_engine = ["GPflow", "PurePython"]
        self.engine = None
        self.input_params = {"date": "", "days_behind": 0, "days_ahead": 0}
        self.obs_date = None

    def select_obs_date(self, date, days_ahead=4, days_behind=4):
        """select observation for a specific date, store as obs_date attribute"""

        assert self.aux is not None, f"aux attribute is None, run load_data() or load_aux_data()"
        assert self.obs is not None, f"obs attribute is None, run load_data() or load_obs_data()"

        # select subset of obs date
        t_range = np.arange(-days_behind, days_ahead + 1)
        date_idx = match(date, self.obs.dims['date']) + t_range
        assert date_idx.min() >= 0, f"date_idx values go negative, found min: {date_idx.min()}"

        select_dims = {"date": self.obs.dims['date'][date_idx]}

        # check if data has already been selected
        # if self.obs_date is not None:
        #     DataDict.dims_equal(self.obs_date['select_dims'], select_dims)

        self.obs_date = self.obs.subset(select_dims=select_dims)
        self.obs_date['date'] = date
        # store the original dates
        self.obs_date['t_to_date'] = self.obs_date.dims['date']
        # change the dates to t
        self.obs_date.set_dim_idx(dim_idx="date", new_idx="t", dim_vals=t_range)
        # dimension used to select from original data
        self.obs_date['select_dims'] = select_dims
        # de-meaned obs
        self.obs_date['de-mean'] = False

    def remove_hold_out_obs_date(self,
                                 hold_out=None,
                                 t=0):
        if hold_out is None:
            print("no hold_out values provided")
            return None
        select_dims = {"sat": hold_out, "t": t}
        self.obs_date.fill_value(fill=np.nan, select_dims=select_dims)

    def prior_mean(self, date, method="fyi_average", **kwargs):

        valid_methods = ["fyi_average", "zero"]
        assert method in valid_methods, f"method: {method} is not in valid_methods: {valid_methods}"

        if method == "fyi_average":
            self._prior_mean_fyi_ave(date, **kwargs)
        elif method == "zero":
            # store as data
            self.mean = DataDict(vals=np.zeros(self.obs.vals.shape[:2]), name="mean")


    def _prior_mean_fyi_ave(self, date, days_behind=9, days_ahead=-1):
        """calculate a trailing mean from first year sea ice data, in line with published paper"""
        date_loc = match(date, self.fyi.dims['date']) + np.arange(-days_behind, days_ahead+1 )
        assert np.min(date_loc) >= 0, f"had negative values in date_loc"

        # select a subset of the data
        _ = self.fyi.subset(select_dims={"date": self.fyi.dims['date'][date_loc]})

        fyi_mean = np.nanmean(_.vals).round(3)
        # store in 2-d array
        fyi_mean = np.full(self.obs.vals.shape[:2], fyi_mean)
        # store as data
        self.mean = DataDict(vals=fyi_mean, name="mean")

    def demean_obs_date(self):
        """subtract mean from obs"""
        # TODO: add a __sub__, __add__ methods to DataDict
        # HACK:
        if self.obs_date.vals.shape[:2] == self.mean.vals.shape[:2]:
            self.obs_date.vals -= self.mean.vals[..., None, None]
        else:
            warnings.warn("mean shape did not align with obs_date, will subtract nan mean")
            self.obs_date.vals -= np.nanmean(self.mean.vals)
        self.obs_date['de-mean'] = True

    def build_kd_tree(self, min_sie=None):
        """build a KD tree using the values from obs_date
        - nans will be removed
        - if min_sie is not None will use to remove locations with insufficient sea ice extent"""

        if min_sie is None:
            sie_bool = True
        else:
            assert isinstance(min_sie, (float, int))
            sie_bool = self.sie.vals >= min_sie
        select_array = (~np.isnan(self.obs_date.vals)) & sie_bool
        _ = self.obs_date.subset(select_array=select_array, new_name="obs_nonan")
        self.X_tree_data = _
        self.X_tree_data['obs_date_dims'] = self.obs_date.dims

        # combine xy data - used for KDtree
        # yx_train = np.array([_['y'], _['x']]).T
        xy_train = np.array([_.dims['x'], _.dims['y']]).T
        # make a KD tree for selecting point
        self.X_tree = scipy.spatial.cKDTree(xy_train)

    def _check_xy_lon_lat(self,
                          x=None,
                          y=None,
                          lon=None,
                          lat=None):

        # require either (x,y) xor (lon,lat) are provided
        assert ((x is None) & (y is None)) ^ ((lon is None) & (lat is None)), \
            f"must supply only (x,y) OR (lon,lat) but not both (or mix of)"

        # if lon, lat were provided, convert to x,y
        if (lon is not None) & (lat is not None):
            if self.verbose:
                print("converting provided (lon,lat) values to (x,y)")
            x, y = WGS84toEASE2_New(lon, lat)
        x, y = to_array(x, y)

        return x, y


    def select_input_output_from_obs_date(self,
                                          x=None,
                                          y=None,
                                          lon=None,
                                          lat=None,
                                          incl_rad=300):
        """get input and output data for a given location"""
        # check x,y / lon,lat inputs (convert to x,y if need be)
        x, y = self._check_xy_lon_lat(x, y, lon, lat)

        # get the points from the input data within radius
        ID = self.X_tree.query_ball_point(x=[x[0], y[0]],
                                          r=incl_rad * 1000)

        # get inputs and outputs for this location
        inputs = np.array([self.X_tree_data.dims[_][ID]
                           for _ in ['x', 'y', 't']]).T

        outputs = self.X_tree_data.vals[ID]

        return inputs, outputs


    def data_select_for_date(self, date, obs=None, days_ahead=4, days_behind=4):
        """given a date, days_ahead and days_behind window
        get arrays of x_train, y_train, t_train, z values
        also sets X_tree attribute (function from scipy.spatial.cKDTree)"""
        # TODO: review this - remove if need be
        # TODO: re
        assert self.aux is not None, f"'aux' attribute is None, run load_aux_data() to populate"

        # TODO:

        # get the dates of the observations
        dates = self.obs['dims']['date']

        if obs is None:
            assert self.obs is not None, f"'obs' attribute is None, run load_obs_data() to populate"
            # get the observation data
            # TODO: is copying needed
            obs = self.obs['data'].copy()
        else:
            if self.verbose:
                print("obs provided (not using obs['data'])")
            # shape check
            for i in [0, 1]:
                assert obs.shape[i] == self.aux['x'].shape[i], \
                    f"provided obs did not match aux data for dimension: {i}  obs: {obs.shape[i]}, aux['x']: {self.aux['x'].shape[i]}"

            assert obs.shape[-1] == len(dates), \
                f"date dimension in obs: {obs.shape[-1]}\ndid not match dates length: {len(dates)}"

        # (x,y) meshgrid coordinates
        xFB = self.aux['x']
        yFB = self.aux['y']

        # need to be explicit with class because DataLoader.data_select is a staticmethod
        # TODO: make data_select a regular method?
        out = super(SeaIceFreeboard, SeaIceFreeboard).data_select(date=date,
                                                                  dates=dates,
                                                                  obs=obs,
                                                                  xFB=xFB,
                                                                  yFB=yFB,
                                                                  days_ahead=days_ahead,
                                                                  days_behind=days_behind)

        self.input_data_all = {
            "x": out[0],
            "y": out[1],
            "t": out[2],
            "z": out[3]
        }
        self.input_params = {"date": date, "days_ahead": days_ahead, "days_behind": days_behind}

        # combine xy data - used for KDtree
        xy_train = np.array([out[0], out[1]]).T
        # make a KD tree for selecting point
        self.X_tree = scipy.spatial.cKDTree(xy_train)

        return out

    def select_data_for_given_date(self,
                                   date,
                                   days_ahead,
                                   days_behind,
                                   hold_out=None,
                                   prior_mean_method="fyi_average",
                                   min_sie=None):

        # select data for a given date (include some days ahead / behind)
        self.select_obs_date(date,
                             days_ahead=days_ahead,
                             days_behind=days_behind)

        # set values on date for hold_out (satellites) to nan
        self.remove_hold_out_obs_date(hold_out=hold_out)

        # calculate the mean for values obs
        self.prior_mean(date,
                        method=prior_mean_method)

        # de-mean the observation (used for the calculation on the given date)
        self.demean_obs_date()

        # build KD-tree
        self.build_kd_tree(min_sie=min_sie)



    def select_data_for_date_location(self, date,
                                      obs=None,
                                      x=None,
                                      y=None,
                                      lon=None,
                                      lat=None,
                                      days_ahead=4,
                                      days_behind=4,
                                      incl_rad=300):

        # check x,y / lon,lat inputs (convert to x,y if need be)
        x, y = self._check_xy_lon_lat(x, y, lon, lat)

        # if (self.obs_date is None)

        # check if need to select new input
        keys = ["date", "days_ahead", "days_behind"]
        params_match = [self.input_params[keys[i]] == _
                        for i, _ in enumerate([date, days_ahead, days_behind])]
        if not all(params_match):

            # select subset of date
            date_loc = match(date, self.obs.dims['date']) + np.arange(-days_behind, days_ahead+1)

            obs_date = obs.subset(select_dims={"date": obs.dims['date'][date_loc]})

            if self.verbose:
                print(f"selecting data for\ndate: {date}\ndays_ahead: {days_ahead}\ndays_behind: {days_behind}")
            self.data_select_for_date(date, obs=obs, days_ahead=days_ahead, days_behind=days_behind)


        else:
            if self.verbose:
                # TODO: make this more clear, i.e. using previously provided data
                print("selecting data using the previously provided values ")

        # get the points from the input data within radius
        ID = self.X_tree.query_ball_point(x=[x, y],
                                          r=incl_rad * 1000)

        # get inputs and outputs for this location
        inputs = np.array([self.input_data_all[_][ID]
                           for _ in ['x', 'y', 't']]).T

        outputs = self.input_data_all["z"][ID]

        return inputs, outputs


    def build_gpr(self,
                  inputs,
                  outputs,
                  # mean=0,
                  length_scales=None,
                  kernel_var=None,
                  likeli_var=None,
                  kernel="Matern32",
                  length_scale_lb=None,
                  length_scale_ub=None,
                  scale_outputs=1.0,
                  scale_inputs=None,
                  engine="GPflow"):

        # TODO: have a check / handle on the valid kernels
        # TOOD: allow for kernels to be provided as objects, rather than just str

        assert engine in self.valid_gpr_engine, f"method: {engine} is not in valid methods" \
                                                f"{self.valid_gpr_engine}"

        # TODO: consider changing observation (y) to z to avoid confusion with x,y used elsewhere?
        self.x = inputs.copy()
        self.y = outputs.copy()

        # require inputs are 2-d
        # TODO: consider if checking shape should be done here,
        if len(self.x.shape) == 1:
            if self.verbose:
                print("inputs was 1-d, broadcasting to make 2-d")
            self.x = self.x[:, None]

        # require outputs are 2-d
        if len(self.y.shape) == 1:
            if self.verbose:
                print("outputs was 1-d, broadcasting to make 2-d")
            self.y = self.y[:, None]

        # de-mean outputs
        # self.mean = mean
        # self.y = self.y - self.mean

        # --
        # apply scaling of inputs and
        # --
        if scale_inputs is None:
            scale_inputs = np.ones(self.x.shape[1])

        # scale_inputs = self._float_list_to_array(scale_inputs)
        # scale_inputs = np.array(scale_inputs) if isinstance(scale_inputs, list) else scale_inputs
        scale_inputs, = to_array(scale_inputs)
        assert len(scale_inputs) == self.x.shape[1], \
            f"scale_inputs did not match expected length: {self.x.shape[1]}"

        self.scale_inputs = scale_inputs
        if self.verbose:
            print(f"scaling inputs by: {scale_inputs}")
        self.x *= scale_inputs

        if scale_outputs is None:
            scale_outputs = np.array([1.0])
        self.scale_outputs = scale_outputs
        if self.verbose:
            print(f"scaling outputs by: {scale_outputs}")
        self.y *= scale_outputs

        # if parameters not provided, set defaults
        if kernel_var is None:
            kernel_var = 1.0
        if likeli_var is None:
            likeli_var = 1.0

        if length_scales is None:
            length_scales = np.ones(inputs.shape[1]) if len(inputs.shape) == 2 else np.array([1.0])
        length_scales = self._float_list_to_array(length_scales)

        if self.verbose:
            print(f"length_scale: {length_scales}")
            print(f"kernel_var: {kernel_var}")
            print(f"likelihood_var: {likeli_var}")

        if engine == "GPflow":
            self.engine = engine
            self._build_gpflow(x=self.x,
                               y=self.y,
                               length_scales=length_scales,
                               kernel_var=kernel_var,
                               likeli_var=likeli_var,
                               length_scale_lb=length_scale_lb,
                               length_scale_ub=length_scale_ub,
                               kernel=kernel)
        elif engine == "PurePython":
            self.engine = engine
            self._build_ppython(x=self.x,
                                y=self.y,
                                length_scales=length_scales,
                                kernel_var=kernel_var,
                                likeli_var=likeli_var,
                                length_scale_lb=length_scale_lb,
                                length_scale_ub=length_scale_ub,
                                kernel=kernel)

    def _build_gpflow(self,
                      x,
                      y,
                      length_scales=1.0,
                      kernel_var=1.0,
                      likeli_var=1.0,
                      length_scale_lb=None,
                      length_scale_ub=None,
                      kernel="Matern32"):

        # require the provide
        assert kernel in gpflow.kernels.__dict__['__all__'], f"kernel provide: {kernel} not value for GPflow"
        # ---
        # kernel
        # ---

        # TODO: needed to determine if these inputs are common across all kernels
        # TODO: should kernel function be set as attribute?
        k = getattr(gpflow.kernels, kernel)(lengthscales=length_scales,
                                            variance=kernel_var)

        # apply constraints, if both supplied
        # TODO: error or warn if both upper and lower not provided
        if (length_scale_lb is not None) & (length_scale_ub is not None):
            # length scale upper bound
            ls_lb = length_scale_lb * self.scale_inputs
            ls_ub = length_scale_ub * self.scale_inputs

            # sigmoid function: to be used for length scales
            sig = tfp.bijectors.Sigmoid(low=tf.constant(ls_lb),
                                        high=tf.constant(ls_ub))
            # TODO: determine if the creation / redefining of the Parameter below requires
            #  - as many parameters as given
            p = k.lengthscales
            k.lengthscales = gpflow.Parameter(p,
                                              trainable=p.trainable,
                                              prior=p.prior,
                                              name=p.name,
                                              transform=sig)
        # ---
        # GPR Model
        # ---

        m = gpflow.models.GPR(data=(x, y),
                              kernel=k,
                              mean_function=None,
                              noise_variance=likeli_var)

        self.model = m

    def _build_ppython(self,
                       x,
                       y,
                       length_scales=1.0,
                       kernel_var=1.0,
                       likeli_var=1.0,
                       length_scale_lb=None,
                       length_scale_ub=None,
                       kernel="Matern32"):

        assert kernel == "Matern32", f"PurePython only has kernel='Matern32' at the moment"

        if length_scale_ub is not None:
            print("length_scale_ub is not handled")
        if length_scale_lb is not None:
            print("length_scale_lb is not handled")

        # hypers = np.concatenate([length_scales, kernel_var, likeli_var])
        # SMLII_mod(hypers, x, y, approx=False, M=None, grad=True)

        pp_gpr = PurePythonGPR(x=x,
                               y=y,
                               length_scales=length_scales,
                               kernel_var=kernel_var,
                               likeli_var=likeli_var)

        self.model = pp_gpr

    def get_hyperparameters(self, scale_hyperparams=False):
        """get the hyper parameters from a GPR model"""
        assert self.engine in self.valid_gpr_engine, f"engine: {self.engine} is not valid"

        if self.engine == "GPflow":

            # length scales
            # TODO: determine here if want to change the length scale names
            #  to correspond with dimension names
            lscale = {f"ls_{self.length_scale_name[i]}": _
                      for i, _ in enumerate(self.model.kernel.lengthscales.numpy())}

            # variances
            kvar = float(self.model.kernel.variance.numpy())
            lvar = float(self.model.likelihood.variance.numpy())

        elif self.engine == "PurePython":

            # length scales
            lscale = {f"ls_{self.length_scale_name[i]}": _
                      for i, _ in enumerate(self.model.length_scales)}

            # variances
            kvar = self.model.kernel_var
            lvar = self.model.likeli_var

        # TODO: need to review this!
        if scale_hyperparams:
            # NOTE: here there is an expectation the keys are in same order as dimension input
            for i, k in enumerate(lscale.keys()):
                lscale[k] /= self.scale_inputs[i]

            kvar /= self.scale_outputs ** 2
            lvar /= self.scale_outputs ** 2

        out = {
            **lscale,
            "kernel_variance": kvar,
            "likelihood_variance": lvar
        }

        return out

    def get_marginal_log_likelihood(self):
        """get the marginal log likelihood"""

        assert self.engine in self.valid_gpr_engine, f"engine: {self.engine} is not valid"

        out = None
        if self.engine == "GPflow":
            # out = {
            #     "mll": self.model.log_marginal_likelihood().numpy()
            # }
            out = self.model.log_marginal_likelihood().numpy()
        elif self.engine == "PurePython":
            out = self.model.get_loglikelihood()

        return out

    def optimise(self, scale_hyperparams=False, **kwargs):
        """optimise the existing (GPR) model"""

        assert self.engine in self.valid_gpr_engine, f"engine: {self.engine} is not valid"

        out = None
        if self.engine == "GPflow":
            opt = gpflow.optimizers.Scipy()

            m = self.model
            opt_logs = opt.minimize(m.training_loss,
                                    m.trainable_variables,
                                    options=dict(maxiter=10000))
            if not opt_logs['success']:
                print("*" * 10)
                print("optimization failed!")
                # TODO: determine if should return None for failed optimisation
                # return None

            # get the hyper parameters, sca
            hyp_params = self.get_hyperparameters(scale_hyperparams=scale_hyperparams)
            mll = self.get_marginal_log_likelihood()
            out = {
                "optimise_success": opt_logs['success'],
                "marginal_loglikelihood": mll,
                **hyp_params
            }

        elif self.engine == "PurePython":

            success = self.model.optimise(**kwargs)
            hyp_params = self.get_hyperparameters(scale_hyperparams=scale_hyperparams)
            mll = self.get_marginal_log_likelihood()
            out = {
                "optimise_success": success,
                "marginal_loglikelihood": mll,
                **hyp_params
            }

        return out

    def get_neighbours_of_grid_loc(self, grid_loc,
                                   coarse_grid_spacing=1,
                                   flatten=True):
        """get x,y location about some grid location"""
        gl0 = grid_loc[0] + np.arange(-coarse_grid_spacing, coarse_grid_spacing + 1)
        gl1 = grid_loc[1] + np.arange(-coarse_grid_spacing, coarse_grid_spacing + 1)

        # trim to be in grid range
        gl0 = gl0[(gl0 >= 0) & (gl0 < self.aux['y'].vals.shape[1])]
        gl1 = gl1[(gl1 >= 0) & (gl1 < self.aux['x'].vals.shape[1])]

        gl0, gl1 = np.meshgrid(gl0, gl1)

        # location to predict on
        x_pred = self.aux['x'].vals[gl0, gl1]
        y_pred = self.aux['y'].vals[gl0, gl1]

        out = [x_pred, y_pred, gl0, gl1]
        if flatten:
            return [_.flatten() for _ in out]
        else:
            return out


    def predict_freeboard(self, x=None, y=None, t=None, lon=None, lat=None):
        """predict freeboard at (x,y,t) or (lon,lat,t) location
        NOTE: t is relative to the window of data available
        """
        # check x,y / lon,lat inputs (convert to x,y if need be)
        x, y = self._check_xy_lon_lat(x, y, lon, lat)

        # assert t is not None, f"t not provided"
        if t is None:
            if self.verbose:
                print("t not provided, getting default")
            # t = self.input_params['days_behind']
            t = np.full(x.shape, 0)

        # make sure x,y,t are arrays
        x, y, t = [self._float_list_to_array(_)
                   for _ in [x, y, t]]
        # which are 2-d (checking only if are 1-d)
        x, y, t = [_[:, None] if len(_.shape) == 1 else _
                   for _ in [x, y, t]]
        # test point
        xs = np.concatenate([x, y, t], axis=1)

        return self.predict(xs)

    def predict(self, xs):
        """generate a prediction for an input (test) point x* (xs"""
        # check inputs - require it to be 2-d array with correct dimension
        # convert if needed
        xs = self._float_list_to_array(xs)
        # check xs shape
        if len(xs.shape) == 1:
            if self.verbose:
                print("xs is 1-d, broadcasting to 2-d")
            xs = xs[None,:]
        assert xs.shape[1] == self.x.shape[1], \
            f"dimension of test point(s): {xs.shape} is not aligned to x/input data: {self.x.shape}"

        # scale input values
        xs *= self.scale_inputs

        out = {}
        if self.engine == "GPflow":
            out = self._predict_gpflow(xs)
            # scale outputs
            # TODO: should this only be applied if
            # out = {k: v * self.scale_outputs ** 2 if re.search("var$", k) else v * self.scale_outputs
            #        for k, v in out.items()}

        elif self.engine == "PurePython":
            out = self._predict_pure_python(xs)

        out['xs'] = xs

        return out

    def _predict_gpflow(self, xs):
        """given a testing input"""
        # TODO: here add mean
        # TODO: do a shape check here
        # xs_ = xs / self.scale_inputs
        y_pred = self.model.predict_y(Xnew=xs)
        f_pred = self.model.predict_f(Xnew=xs)

        # TODO: add mean and scale?
        out = {
            "f*": f_pred[0].numpy()[:, 0],
            "f*_var": f_pred[1].numpy()[:, 0],
            "y": y_pred[0].numpy()[:, 0],
            "y_var": y_pred[1].numpy()[:, 0],
        }
        return out

    def _predict_pure_python(self, xs, **kwargs):
        # NOTE: is it expected the data (self.y) has been de-meaned already
        # adding the mean back should happen else where
        return self.model.predict(xs, mean=0, **kwargs)


    def _float_list_to_array(self, x):
        # TODO: let _float_list_to_array just call to_array
        if isinstance(x, (float, int)):
            return np.array([x / 1.0])
        elif isinstance(x, list):
            return np.array(x, dtype=float)
        else:
            return x


    def sat_obs_location_on_date(self,
                                 date,
                                 sat_names=None):
        """given a date return a bool array specifying the locations where
        satellite observations exists (satellite names specified in sat_names)"""
        if sat_names is None:
            sat_names = []

        assert isinstance(sat_names, (list, tuple, np.ndarray)), f"sat_names should be list, tuple or ndarray"

        # check provided sat names and dates are valid
        for sn in sat_names:
            assert sn in self.obs.dims['sat'], f"sat_name: {sn} not valid, must be in: {self.obs['dims']['sat']}"

        assert date in self.obs.dims['date'], f"date: {date} is not in obs['dims']['date']"

        sat_obs_loc_bool = np.zeros(self.obs.vals.shape[:2], dtype=bool)

        if len(sat_names):
            if self.verbose > 1:
                print(f"identifying sat. observations for date: {date} and satellites"
                      f"{sat_names}")

            # copy observation data
            # - so can set hold_out data to np.nan
            # obs = self.obs.vals#.copy()

            for sn in sat_names:
                _ = self.obs.subset({"date": date, "sat": sn})

                # get the location of the hold_out (sat)
                # sat_loc = np.in1d(self.obs.dims['sat'], sn)
                # date_loc = np.in1d(self.obs.dims['date'], date)
                # get hold_out data observations locations
                # sat_obs_loc_bool[~np.isnan(obs[:, :, sat_loc, date_loc][..., 0])] = True

                sat_obs_loc_bool[~np.isnan(_.vals[..., 0, 0])] = True

        return sat_obs_loc_bool


    def select_gp_locations(self,
                            date=None,
                            min_sie=0.15,
                            coarse_grid_spacing=1,
                            grid_space_offset=0,
                            sat_names=None):
        """
        get a bool array of the locations to calculate GP
        - only where sie exists (not nan)
        - sie >= min_sie
        - on coarse grid points (coarse_grid_spacing=1 will take all)
        - on locations of satellite observations for date if sat_names != None
        """
        # TODO: review / clean up select_gp_locations method
        if date is None:
            date = self.obs_date['date']

        assert self.sie is not None, f"require sie attribute to specified"
        sie = self.sie.vals #['data']
        sie_dates = self.sie.dims['date'] #['dims']['date']

        # default will be to calculate GPs for all points
        select_bool = np.ones(sie.shape[:2], dtype=bool)

        assert date in sie_dates, f"date: {date} is not in sie['dims']['date']"
        # dloc = np.where(np.in1d(sie_dates, date))[0][0]
        dloc = match(date, sie_dates)[0]

        # exclude points where there is now sea ice extent
        select_bool[np.isnan(sie[..., dloc])] = False

        # exclude points where there is insufficient sie
        select_bool[sie[..., dloc] < min_sie] = False

        # coarse grid
        cgrid = self.coarse_grid(coarse_grid_spacing,
                                 grid_space_offset=grid_space_offset,
                                 x_size=sie.shape[1],
                                 y_size=sie.shape[0])

        select_bool = select_bool & cgrid

        # only on satellite locations for the day
        if sat_names is not None:
            select_bool = select_bool & self.sat_obs_location_on_date(date, sat_names)

        return select_bool


    def run(self,
            date,
            season="2018-2019",
            days_ahead=4,
            days_behind=4,
            incl_rad=300,
            engine="GPflow",
            kernel="Matern32",
            coarse_grid_spacing=1,
            min_sie=0.15,
            hold_out=None,
            pred_on_hold_out=True,
            load_data=True,
            aux_data_dir=None,
            sat_data_dir=None):
        """
        wrapper function to run optimal interpolation of sea ice freeboard for a given date
        """

        # TODO: SeaIceFreeboard run methods needs to be completed
        if load_data:
            # can load data via inherited DataLoader methods
            aux_data_dir = aux_data_dir if aux_data_dir is not None else get_data_path("aux")
            self.load_aux_data(aux_data_dir=aux_data_dir,
                               season=season)

            sat_data_dir = sat_data_dir if sat_data_dir is not None else get_data_path("CS2S3_CPOM")
            self.load_obs_data(sat_data_dir=sat_data_dir,
                               season=season)

        assert self.aux is not None, f"aux attribute, run load_aux_data(), or set load_data=True"
        assert self.obs is not None, f"obs attribute, run load_ons_data(), or set load_data=True"

        # ---
        # locations to predict / build GP model
        # ---

        select_bool = self.select_gp_locations(
                            date=date,
                            min_sie=min_sie,
                            coarse_grid_spacing=coarse_grid_spacing,
                            grid_space_offset=0,
                            sat_names=hold_out)

        # ---
        # remove obs from hold_out sat
        # ---

        # TODO: wrap this up into a method?
        obs = self.obs['data'].copy()
        if hold_out is not None:
            if self.verbose:
                print(f"removing observations from: {hold_out} for date: {date}")

            for ho in hold_out:
                print(f"removing: {ho} data")
                # get the location of the hold_out (sat)
                sat_loc = np.in1d(self.obs['dims']['sat'], ho)
                date_loc = np.in1d(self.obs['dims']['date'], date)
                # set the observations at the hold out location to nan
                obs[:, :, sat_loc, date_loc] = np.nan

        # ----
        # de-mean:
        # ----

        # TODO: should have a prior mean method, should return an array aligned (broadcast-able)
        #  with obs so can easily de-mean
        # TODO: wrap getting first year sea ice into a method, make it part of aux data, or its own?

        datapath = get_data_path()
        # TODO: make sure dates are aligned to satellite data, read in same way
        cs2_FYI = np.load(
            datapath + f'/aux/CS2_{self.grid_res}_FYI_20181101-20190428.npy')
        # create an array of dates
        cs2_FYI_dates = np.arange(np.datetime64("2018-11-01"), np.datetime64("2019-04-29"))
        cs2_FYI_dates = np.array([re.sub("-", "", i) for i in cs2_FYI_dates.astype(str)])

        # mean = np.nanmean(cs2_FYI[..., (day - days_behind):(day + days_ahead + 1)]).round(4)
        # TODO: should have checks that range is valid here
        print("using CS2_FYI data for prior mean")
        cday = np.where(np.in1d(cs2_FYI_dates, date))[0][0]
        # TODO: should this be trailing 31 days?
        # mean = np.nanmean(cs2_FYI[..., (cday - days_behind):(cday + days_ahead + 1)]).round(4)
        mean = np.nanmean(cs2_FYI[..., (cday - (days_behind + days_ahead + 1)):cday]).round(4)

        # ----
        # calculate a GP for each location in select_bool
        # ----

        num_loc = select_bool.sum()
        select_loc = np.where(select_bool)

        # for each location
        for i in range(num_loc):

            if (i % 100) == 0:
                print("*" * 75)
                print(f"{i + 1}/{num_loc + 1}")

            # ----
            # get the input and output data for model
            # ----

            grid_loc = select_loc[0][i], select_loc[1][i]
            x_ = self.aux['x'][grid_loc]
            y_ = self.aux['y'][grid_loc]

            # TODO: here could allow for obs to be a dict of the subset of all, which could then be
            #  - have mean removed?

            inputs, outputs = self.select_data_for_date_location(date=date,
                                                                 obs=obs,
                                                                 x=x_,
                                                                 y=y_,
                                                                 days_ahead=days_ahead,
                                                                 days_behind=days_behind,
                                                                 incl_rad=incl_rad)

            # ---
            # build a GPR model for data
            # ---

            sifb.build_gpr(inputs=inputs,
                           outputs=outputs - mean,
                           scale_inputs=[1 / (grid_res * 1000), 1 / (grid_res * 1000), 1.0],
                           # scale_outputs=1 / 100,
                           engine=engine,
                           # mean=0,
                           # length_scales=None,
                           # kernel_var=None,
                           # likeli_var=None,
                           # kernel="Matern32",
                           # length_scale_lb=None,
                           # length_scale_ub=None,
                           # scale_outputs=1.0,
                           )

            # ---
            # get the hyper parameters
            # ---

            hps = sifb.get_hyperparameters(scale_hyperparams=False)

            # ---
            # optimise model
            # ---

            # key-word arguments for optimisation (used by PurePython implementation)
            kwargs = {}
            if engine == "PurePython":
                kwargs = {
                    "jac": True,
                    "opt_method": "CG"
                }
                # TODO: determine what is the preferable optimisation parameters (kwargs)
                # kwargs = {
                #     "jac": False,
                #     "opt_method": "L-BFGS-B"
                # }

            opt_hyp = sifb.optimise(scale_hyperparams=False, **kwargs)
            res[engine]["opt_hyp"] = opt_hyp

            # ---
            # make predictions
            # ---

            preds = sifb.predict_freeboard(lon=lon, lat=lat)
            res[engine]["pred"] = preds


if __name__ == "__main__":

    from OptimalInterpolation import get_data_path

    # ---
    # parameters
    # ---

    season = "2018-2019"
    grid_res = 50
    date = "20181203"
    days_ahead = 4
    days_behind = 4

    # radius to include - in km
    incl_rad = 300

    # location
    # can either specify x,y or lon,lat
    # x, y = -212500.0, -862500.0
    lon, lat = -13.84069549, 82.040178

    # --
    # initialise SeaIceFreeboard class
    # --

    sifb = SeaIceFreeboard(grid_res=f"{grid_res}km",
                           length_scale_name=['x', 'y', 't'])

    # ---
    # read / load data
    # ---

    sifb.load_data(aux_data_dir=get_data_path("aux"),
                   sat_data_dir=get_data_path("CS2S3_CPOM"),
                   season=season)

    # ---
    # select data for a given date and location
    # ---

    # TODO: create a method to select_inputs_outputs_from_date_location
    #  - and wrap the following lines up

    sifb.select_data_for_given_date(date=date,
                                    days_ahead=days_ahead,
                                    days_behind=days_behind,
                                    hold_out=None,
                                    prior_mean_method="fyi_average",
                                    min_sie=None)

    # select inputs for a given location
    inputs, outputs = sifb.select_input_output_from_obs_date(lon=lon,
                                                             lat=lat,
                                                             incl_rad=incl_rad)

    # ---
    # build GPR mode, optimise and predict
    # ---

    # using different 'engines' (backends)
    engines = ["GPflow", "PurePython"]

    res = {}
    for engine in engines:
        print("*" * 20)
        print(engine)
        res[engine] = {}

        # ---
        # build a GPR model for data
        # ---

        sifb.build_gpr(inputs=inputs,
                       outputs=outputs,
                       scale_inputs=[1 / (grid_res * 1000), 1 / (grid_res * 1000), 1.0],
                       # scale_outputs=1 / 100,
                       engine=engine)

        # ---
        # get the hyper parameters
        # ---

        hps = sifb.get_hyperparameters(scale_hyperparams=False)

        # ---
        # optimise model
        # ---

        # key-word arguments for optimisation (used by PurePython implementation)
        kwargs = {}
        if engine == "PurePython":
            #
            kwargs = {
                "jac": True,
                "opt_method": "CG"
            }
            # TODO: determine what is the preferable optimisation parameters (kwargs)
            # kwargs = {
            #     "jac": False,
            #     "opt_method": "L-BFGS-B"
            # }

        opt_hyp = sifb.optimise(scale_hyperparams=False, **kwargs)
        res[engine]["opt_hyp"] = opt_hyp

        # ---
        # make predictions
        # ---

        preds = sifb.predict_freeboard(lon=lon, lat=lat)
        res[engine]["pred"] = preds

    # ----
    # compare values
    # ----

    # compare the values from the engines
    for i, ei in enumerate(engines):
        for j in range(i + 1, len(engines)):
            ej = engines[j]
            print("-" * 50)
            print(f"Engines: {ei} vs {ej}")
            for k, v in res[ei].items():
                print("*" * 25)
                print(k)
                for kk, vv in v.items():
                    print("-" * 10)
                    print(kk)
                    print(f"{ei:<10}:\t\t{vv}")
                    print(f"{ej:<10}:\t\t{res[ej][k][kk]}")
                    print(f"{'diff':<10}:\t\t{vv - res[ej][k][kk]}")

