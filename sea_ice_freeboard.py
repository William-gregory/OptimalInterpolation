# Sea Ice Freeboard class - for interpolating using Gaussian Process Regression
import re
import gpflow
import numpy as np
import scipy

import tensorflow as tf
import tensorflow_probability as tfp

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

    def __init__(self, grid_res="25km", sat_list=None, verbose=True):

        super().__init__(grid_res=grid_res, sat_list=sat_list, verbose=verbose)

        self.parameters = None
        self.mean = None
        self.inputs = None
        self.outputs = None
        self.scale_outputs = None
        self.scale_inputs = None
        self.model = None
        self.X_tree = None
        self.input_data_all = None
        self.valid_gpr_engine = ["GPflow", "PurePython"]
        self.engine = None
        self.input_params = {"date": "", "days_behind": 0, "days_ahead": 0}

    def data_select_for_date(self, date, days_ahead=4, days_behind=4):
        """given a date, days_ahead and days_behind window
        get arrays of x_train, y_train, t_train, z values
        also sets X_tree attribute (function from scipy.spatial.cKDTree)"""

        assert self.obs is not None, f"'obs' attribute is None, run load_obs_data() to populate"
        assert self.aux is not None, f"'aux' attribute is None, run load_aux_data() to populate"

        # get the observation data
        obs = self.obs['data']

        # get the dates of the observations
        dates = self.obs['dims']['date']

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

        return x, y

    def select_data_for_date_location(self, date,
                                      x=None,
                                      y=None,
                                      lon=None,
                                      lat=None,
                                      days_ahead=4,
                                      days_behind=4,
                                      incl_rad=300):

        # check x,y / lon,lat inputs (convert to x,y if need be)
        x, y = self._check_xy_lon_lat(x, y, lon, lat)

        # check if need to select new input
        keys = ["date", "days_ahead", "days_behind"]
        params_match = [self.input_params[keys[i]] == _
                        for i, _ in enumerate([date, days_ahead, days_behind])]
        if not all(params_match):
            if self.verbose:
                print(f"selecting data for\ndate: {date}\ndays_ahead: {days_ahead}\ndays_behind: {days_behind}")
            self.data_select_for_date(date, days_ahead=days_ahead, days_behind=days_behind)

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
                  mean=0,
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
        self.mean = mean
        self.y = self.y - self.mean

        # --
        # apply scaling of inputs and
        # --
        if scale_inputs is None:
            scale_inputs = np.ones(self.x.shape[1])

        scale_inputs = self._float_list_to_array(scale_inputs)
        # scale_inputs = np.array(scale_inputs) if isinstance(scale_inputs, list) else scale_inputs
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
            # sigmoid function: to be used for length scales
            sig = tfp.bijectors.Sigmoid(low=tf.constant(length_scale_lb),
                                        high=tf.constant(length_scale_ub))
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
            lscale = {f"ls_{i}": _
                      for i, _ in enumerate(self.model.kernel.lengthscales.numpy())}

            # variances
            kvar = float(self.model.kernel.variance.numpy())
            lvar = float(self.model.likelihood.variance.numpy())

        elif self.engine == "PurePython":

            # length scales
            lscale = {f"ls_{i}": _
                      for i, _ in enumerate(self.model.length_scales)}

            # variances
            kvar = self.model.kernel_var
            lvar = self.model.likeli_var

        # TODO: need to review this!
        if scale_hyperparams:
            # NOTE: here there is an expectation the keys are in same order as dimension input
            for i, k in enumerate(lscale.keys()):
                lscale[k] *= self.scale_inputs[i]

            kvar *= self.scale_outputs ** 2
            lvar *= self.scale_outputs ** 2

        out = {
            **lscale,
            "kernel_variance": kvar,
            "likelihood_variance": lvar
        }

        return out

    def get_marginal_log_likelihood(self, **kwargs):
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

            # %%
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
            t = self.input_params['days_behind']

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

        return self.model.predict(xs, mean=self.mean, **kwargs)


    def _float_list_to_array(self, x):
        if isinstance(x, (float, int)):
            return np.array([x / 1.0])
        elif isinstance(x, list):
            return np.array(x, dtype=float)
        else:
            return x


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

    sifb = SeaIceFreeboard(grid_res=f"{grid_res}km")

    # ---
    # read / load data
    # ---

    # can load data via inherited DataLoader methods
    sifb.load_aux_data(aux_data_dir=get_data_path("aux"),
                       season=season)
    sifb.load_obs_data(sat_data_dir=get_data_path("CS2S3_CPOM"),
                       season=season)

    # ---
    # select data for a given location and date
    # ---

    inputs, outputs = sifb.select_data_for_date_location(date=date,
                                                         lon=lon,
                                                         lat=lat,
                                                         days_ahead=days_ahead,
                                                         days_behind=days_behind,
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




    # get the values to compare
    key_vals = {k: list(res[engines[0]][k].keys()) for k in res[engines[0]].keys()}

    for k, v in key_vals.items():

        for _ in v:
            [res[e][k][_] for e in engines]
        break

