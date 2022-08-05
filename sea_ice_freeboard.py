# Sea Ice Freeboard class - for interpolating using Gaussian Process Regression
import re
import os
import json
import copy
import gpflow
import numpy as np
import pandas as pd
import scipy
import warnings
import datetime
import time
import subprocess
import shelve
from ast import literal_eval as make_tuple

from scipy.stats import shapiro, norm

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.client import device_lib

from OptimalInterpolation import get_data_path
from OptimalInterpolation.data_dict import DataDict, match, to_array
from OptimalInterpolation.data_loader import DataLoader
from OptimalInterpolation.utils import WGS84toEASE2_New, \
    EASE2toWGS84_New, SMLII_mod, SGPkernel, GPR, get_git_information, move_to_archive

from gpflow.mean_functions import Constant
from gpflow.utilities import parameter_dict, multiple_assign


# TODO: put PurePythonGPR in separate script / module
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
            "f*_var": res[1] ** 2,
            "y": res[0].flatten(),
            "y_var": res[1] ** 2 + self.likeli_var
        }
        return out


class SeaIceFreeboard(DataLoader):
    mean_functions = {
        "constant": Constant()
    }

    def __init__(self, grid_res="25km", sat_list=None, verbose=True,
                 length_scale_name=None,
                 rng_seed=42):

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
        self.valid_gpr_engine = ["GPflow", "PurePython", "GPflow_svgp"]
        self.engine = None
        self.input_params = {"date": "", "days_behind": 0, "days_ahead": 0}
        self.obs_date = None


        # random number generator - used for selecting inducing points
        self.rnd = np.random.default_rng(seed=rng_seed)

        # ----
        # devices - get info on GPU
        # ---

        dev = device_lib.list_local_devices()
        gpu_name = None
        for d in dev:
            # check if device is GPU
            # - will break after first
            if d.device_type == "GPU":
                print("found GPU")
                try:
                    name_loc = re.search("name:(.*?),", d.physical_device_desc).span(0)
                    gpu_name = d.physical_device_desc[(name_loc[0] + 6):(name_loc[1] - 1)]
                except Exception as e:
                    print("there was some issue getting GPU name")
                    print(e)
                break
        self.gpu_name = gpu_name

    def select_obs_date(self, date, days_ahead=4, days_behind=4):
        """select observation for a specific date, store as obs_date attribute"""

        assert self.aux is not None, f"aux attribute is None, run load_data() or load_aux_data()"
        assert self.obs is not None, f"obs attribute is None, run load_data() or load_obs_data()"

        if self.raw_obs is None:
            print("selecting data from gridded observations (raw_obs is None)")
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
            self.obs_date['raw_data'] = False

        else:
            # TODO: tidy select_obs_date for raw_obs up
            # convert date to datetime64 - so can work with datetime in dims
            date_ = np.datetime64(datetime.datetime.strptime(date, "%Y%m%d"))
            date_ = date_.astype('datetime64[s]')
            # get the interval of datetime to select
            # NOTE: here it is assumed dates are continuous (i.e. there are no gaps)
            date_behind = date_ - np.timedelta64(days_behind, 'D')
            # NOTE: for date ahead adding one because the day start at 00:00
            date_ahead = date_ + np.timedelta64(days_ahead + 1, 'D')

            select_array = (self.raw_obs.dims['datetime'] >= date_behind) & (self.raw_obs.dims['datetime'] < date_ahead)

            self.obs_date = self.raw_obs.subset(select_array=select_array)

            # TODO: review the following to determine how many of the obs_date keys are needed
            # TODO: determine if this date can be datetim64
            self.obs_date['date'] = date
            self.obs_date['t_to_date'] = self.obs_date.dims['datetime']

            # get the difference in terms of fraction of a day
            # - datetimes are in seconds, so divide by seconds in a day
            t_range = (self.obs_date.dims['datetime'] - date_).astype(float) / (60 * 60 * 24)
            # TODO: consider if this should be done else where
            warnings.warn("offsetting t_range by 0.5, so interval is symmetrical, really should be"
                          "making predictions at noon")
            t_range -= 0.5
            self.obs_date.set_dim_idx(dim_idx="datetime", new_idx="t", dim_vals=t_range)

            # dimension used to select from original data
            # self.obs_date['select_array'] = select_array
            # de-meaned obs
            self.obs_date['de-mean'] = False
            self.obs_date['raw_data'] = True


    def remove_hold_out_obs_date(self,
                                 hold_out=None,
                                 t=0):
        if hold_out is None:
            print("no hold_out values provided")
            return None
        if self.obs_date['raw_data']:
            warnings.warn("removing observation on hold_out using time adjust by 0.5")
            select_array = np.in1d(self.obs_date.dims['sat'], hold_out) & \
                           (self.obs_date.dims['t'] < 0.5) & \
                           (self.obs_date.dims['t'] >= -0.5)
            self.obs_date['held_out'] = self.obs_date.subset(select_array=select_array).copy()
            self.obs_date.fill_value(fill=np.nan, select_array=select_array)
        else:
            select_dims = {"sat": hold_out, "t": t}
            self.obs_date.fill_value(fill=np.nan, select_dims=select_dims)

    def prior_mean(self, date, method="fyi_average", **kwargs):

        valid_methods = ["fyi_average", "zero", "demean_outputs"]
        assert method in valid_methods, f"method: {method} is not in valid_methods: {valid_methods}"

        if method == "fyi_average":
            self._prior_mean_fyi_ave(date, **kwargs)
        elif method == "zero":
            # store as data
            # self.mean = DataDict(vals=np.zeros(self.obs.vals.shape[:2]), name="mean", default_dim_name="grid_loc_")
            self.mean = DataDict.full(shape=self.obs.vals.shape[:2],
                                      fill_val=0.,
                                      name="mean",
                                      default_dim_name="grid_loc_")
        elif method == "demean_outputs":
            # TODO: this should calculate the mean of the inputs for the corresponding:
            #  - days_ahead, days_behind and radius. currently such action is done outside of this function
            #  - where, also, the mean values are set. so for now set these values to 0, so won't impact inputs when
            #  - they're subtracted
            # TODO: need to reconsider the above, demean_obs_date() will subtract mean from each location (and each date)
            #  - where what want to do is the above - i.e. subtract data from inputs such their mean is 0
            #  - which means the same input will have different mean subtracted, as the mean will depend on
            #  - other values in input (set)
            self.mean = DataDict.full(shape=self.obs.vals.shape[:2],
                                      fill_val=0.,
                                      name="mean",
                                      default_dim_name="grid_loc_")

    def _prior_mean_fyi_ave(self, date, fyi_days_behind=9, fyi_days_ahead=-1):
        """calculate a trailing mean from first year sea ice data, in line with published paper"""
        date_loc = match(date, self.fyi.dims['date']) + np.arange(-fyi_days_behind, fyi_days_ahead + 1)
        assert np.min(date_loc) >= 0, f"had negative values in date_loc"

        # select a subset of the data
        _ = self.fyi.subset(select_dims={"date": self.fyi.dims['date'][date_loc]})

        fyi_mean = np.nanmean(_.vals).round(3)
        # store in 2-d array
        fyi_mean = np.full(self.obs.vals.shape[:2], fyi_mean)
        # store as data
        self.mean = DataDict(vals=fyi_mean, name="mean", default_dim_name="grid_loc_")

    def demean_obs_date(self):
        """subtract mean from obs"""
        # TODO: could use subtraction on DataDict for this
        # HACK: want to subtract mean value from each point, for fyi_mean all means are the same (for given date)
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

        # TODO: move removing nans into remove_hold_out_obs_date()
        select_array = (~np.isnan(self.obs_date.vals))
        if not self.obs_date['raw_data']:
            select_array = select_array & sie_bool
        _ = self.obs_date.subset(select_array=select_array, new_name="obs_nonan")
        self.X_tree_data = _
        # TODO: is X_tree_data['obs_date_dims'] used again?
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
            date_loc = match(date, self.obs.dims['date']) + np.arange(-days_behind, days_ahead + 1)

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
                  mean_function=None,
                  engine="GPflow",
                  min_obs_for_svgp=1000,
                  **inducing_point_params):

        # TOD0: length scales and variances should be scaled in the same way as inputs /outputs
        # TODO: have a check / handle on the valid kernels
        # TODO: allow for kernels to be provided as objects, rather than just str

        assert engine in self.valid_gpr_engine, f"method: {engine} is not in valid methods" \
                                                f"{self.valid_gpr_engine}"

        # min_obs_for_svgp = inducing_point_params.get('min_obs_for_svgp', 1000)

        # TODO: consider changing observation (y) to z to avoid confusion with x,y used elsewhere?
        self.x = inputs.copy()
        self.y = outputs.copy()

        # require inputs are 2-d
        # TODO: consider if checking shape should be done here,
        if len(self.x.shape) == 1:
            if self.verbose > 3:
                print("inputs was 1-d, broadcasting to make 2-d")
            self.x = self.x[:, None]

        # require outputs are 2-d
        if len(self.y.shape) == 1:
            if self.verbose > 3:
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
        if self.verbose > 3:
            print(f"scaling inputs by: {scale_inputs}")
        self.x *= scale_inputs

        if scale_outputs is None:
            scale_outputs = np.array([1.0])
        self.scale_outputs = scale_outputs
        if self.verbose > 3:
            print(f"scaling outputs by: {scale_outputs}")
        self.y *= scale_outputs

        # if parameters not provided, set defaults
        if kernel_var is None:
            kernel_var = 1.0
        if likeli_var is None:
            likeli_var = 1.0

        if length_scales is None:
            length_scales = np.ones(inputs.shape[1]) if len(inputs.shape) == 2 else np.array([1.0])
        # length_scales = self._float_list_to_array(length_scales)
        length_scales, = to_array(length_scales)

        if self.verbose > 3:
            print("initial hyper parameter values")
            print(f"length_scale: {length_scales}")
            print(f"kernel_var: {kernel_var}")
            print(f"likelihood_var: {likeli_var}")

        # if the number of observations is to few Stochastic Variational

        if (engine == "GPflow_svgp") & (len(self.x) <= min_obs_for_svgp):
            if self.verbose > 1:
                print("too few entries for 'GPflow_svgp', will use 'GPflow'")
            engine = "GPflow"

        if engine == "GPflow":
            self.engine = engine
            self._build_gpflow(x=self.x,
                               y=self.y,
                               length_scales=length_scales,
                               kernel_var=kernel_var,
                               likeli_var=likeli_var,
                               length_scale_lb=length_scale_lb,
                               length_scale_ub=length_scale_ub,
                               mean_function=mean_function,
                               kernel=kernel)

        elif engine == "GPflow_svgp":
            self.engine = engine

            # NOTE: y is not used
            self._build_gpflow_svgp(x=self.x,
                                    y=self.y,
                                    length_scales=length_scales,
                                    kernel_var=kernel_var,
                                    likeli_var=likeli_var,
                                    length_scale_lb=length_scale_lb,
                                    length_scale_ub=length_scale_ub,
                                    mean_function=mean_function,
                                    kernel=kernel,
                                    **inducing_point_params)

        elif engine == "PurePython":
            self.engine = engine

            assert mean_function is None, "mean_function is not None and engine='PurePython', this isn't handled"
            self._build_ppython(x=self.x,
                                y=self.y,
                                length_scales=length_scales,
                                kernel_var=kernel_var,
                                likeli_var=likeli_var,
                                length_scale_lb=length_scale_lb,
                                length_scale_ub=length_scale_ub,
                                kernel=kernel)

    def _get_kernel_mean_function(self,
                                  length_scales=1.0,
                                  kernel_var=1.0,
                                  length_scale_lb=None,
                                  length_scale_ub=None,
                                  kernel="Matern32",
                                  mean_function=None
                                  ):
        """get kernel and mean_function which can be used by gpflow.models.GPR (_build_gpflow)
        or gpflow.models.SVGP (_build_gpflow_svgp)
        """
        # require the provide
        assert kernel in gpflow.kernels.__dict__['__all__'], f"kernel provide: {kernel} not value for GPflow"

        # ---
        # mean function
        # ---

        if mean_function is not None:

            assert mean_function in self.mean_functions, \
                f"mean_function: {mean_function} not in a valid option:\n" \
                f"{list(self.mean_functions.keys())}"

            # NOTE: using the class attribute likely will use the same object
            # - which lead the optimizer failing?
            if mean_function == 'constant':
                mean_function = Constant(c=np.array([np.mean(self.y)]))
            # mean_function = self.mean_functions[mean_function]

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

            # check if length scales are at bounds - move them off if they are
            ls_scales = k.lengthscales.numpy()
            if (ls_scales == ls_lb).any():
                ls_scales[ls_scales == ls_lb] = ls_ub[ls_scales == ls_lb] + 1e-6
            if (ls_scales == ls_ub).any():
                ls_scales[ls_scales == ls_ub] = ls_ub[ls_scales == ls_ub] - 1e-6

            # if the length scale values have changed then assign the new values
            if (k.lengthscales.numpy() != ls_scales).any():
                k.lengthscales.assign(ls_scales)
            p = k.lengthscales

            k.lengthscales = gpflow.Parameter(p,
                                              trainable=p.trainable,
                                              prior=p.prior,
                                              name=p.name,
                                              transform=sig)

        return k, mean_function

    def _build_gpflow(self,
                      x,
                      y,
                      length_scales=1.0,
                      kernel_var=1.0,
                      likeli_var=1.0,
                      length_scale_lb=None,
                      length_scale_ub=None,
                      kernel="Matern32",
                      mean_function=None):

        # get the kernel and mean function
        k, mean_function = self._get_kernel_mean_function(length_scales=length_scales,
                                                          kernel_var=kernel_var,
                                                          length_scale_lb=length_scale_lb,
                                                          length_scale_ub=length_scale_ub,
                                                          kernel=kernel,
                                                          mean_function=mean_function)

        # ---
        # GPR Model
        # ---

        m = gpflow.models.GPR(data=(x, y),
                              kernel=k,
                              mean_function=mean_function,
                              noise_variance=likeli_var)

        self.model = m

    def _build_gpflow_svgp(self,
                           x,
                           y,
                           length_scales=1.0,
                           kernel_var=1.0,
                           likeli_var=1.0,
                           length_scale_lb=None,
                           length_scale_ub=None,
                           kernel="Matern32",
                           mean_function=None,
                           inducing_variable=None,
                           num_inducing_points=None,
                           **kwargs):

        # get the kernel and mean function
        k, mean_function = self._get_kernel_mean_function(length_scales=length_scales,
                                                          kernel_var=kernel_var,
                                                          length_scale_lb=length_scale_lb,
                                                          length_scale_ub=length_scale_ub,
                                                          kernel=kernel,
                                                          mean_function=mean_function)

        # ---
        # Stochastic Variational GP Model
        # ---

        if inducing_variable is not None:
            # TODO: check inducing_variable has the correct properities
            Z = inducing_variable

        else:
            if self.verbose > 1:
                print("no inducing_variable provided, will pick ")
            assert num_inducing_points is not None, f"num_inducing_points must be specified if inducing_variable is not"
            if num_inducing_points > len(x):
                print(f"num_inducing_points={num_inducing_points} is greater than\nlen(x)={len(x)}\nwill change to num_inducing_points")
                num_inducing_points = len(x)

            M_loc = self.rnd.choice(np.arange(len(x)), num_inducing_points, replace=False)
            Z = x[M_loc, :]

        N = len(x)
        m = gpflow.models.SVGP(kernel=k,
                               likelihood=gpflow.likelihoods.Gaussian(variance=likeli_var),
                               inducing_variable=Z,
                               mean_function=mean_function,
                               num_data=N)

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

        mean_func_params = {}
        if self.engine in ["GPflow", "GPflow_svgp"]:
            # TODO: if model has mean_function attribute specified, get parameter
            # length scales
            # TODO: determine here if want to change the length scale names
            #  to correspond with dimension names
            lscale = {f"ls_{self.length_scale_name[i]}": _
                      for i, _ in enumerate(self.model.kernel.lengthscales.numpy())}

            # variances
            kvar = float(self.model.kernel.variance.numpy())
            lvar = float(self.model.likelihood.variance.numpy())

            # check for mean_function parameters
            if self.model.mean_function.name != "zero":

                if self.model.mean_function.name == "constant":
                    mean_func_params["mean_func"] = self.model.mean_function.name
                    mean_func_params["mean_func_c"] = float(self.model.mean_function.c.numpy())
                else:
                    warnings.warn(f"mean_function.name: {self.model.mean_function.name} not understood")

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

            # TODO: double check scaling of mean function values
            for k in mean_func_params.keys():
                if isinstance(mean_func_params[k], (int, float)):
                    mean_func_params[k] /= self.scale_outputs

        out = {
            **lscale,
            "kernel_variance": kvar,
            "likelihood_variance": lvar,
            **mean_func_params
        }

        return out

    def get_marginal_log_likelihood(self):
        """get the marginal log likelihood"""

        assert self.engine in self.valid_gpr_engine, f"engine: {self.engine} is not valid"

        out = None
        if self.engine == "GPflow":
            out = self.model.log_marginal_likelihood().numpy()
        elif self.engine == "GPflow_svgp":
            # this is the ELBO - evidence lower bound on log likelihood
            out = self.model.maximum_log_likelihood_objective((self.x, self.y)).numpy()
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

        elif self.engine == "GPflow_svgp":

            # TODO: store (mini batch) elbo values (the objective function) some where
            opt_logs = self._svgp_optimise(**kwargs)

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

    def _svgp_run_adam(self, model, iterations, train_dataset, minibatch_size,
                       early_stop=True, persistance=100,
                       save_best=False):
        """
        Utility function running the Adam optimizer

        :param model: GPflow model
        :param interations: number of iterations
        """
        # TODO: REMOVE _svgp_run_adam if it's not being used
        # based off of:
        # https://gpflow.github.io/GPflow/develop/notebooks/advanced/gps_for_big_data.html

        # TODO: in this method should the full elbo be evaluated
        #  - i.e. using full dataset
        if save_best:
            warnings.warn(f"save_best: {save_best}, this will likely slow process")

        # REMOVE THIS?
        # elbo = tf.function(model.elbo)

        # Create an Adam Optimizer action
        logf = []
        train_iter = iter(train_dataset.batch(minibatch_size))


        training_loss = model.training_loss_closure(train_iter, compile=True)
        # TODO: add parameters for Adam
        optimizer = tf.optimizers.Adam()

        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, model.trainable_variables)

        # initialise the maximum elbo
        max_elbo = -np.inf
        best_step = 0
        stopped_early = False
        # NOTE: need to use copy.deepcopy() - this could slow things down
        params = parameter_dict(model)

        for step in range(iterations):
            optimization_step()
            if step % 10 == 0:
                # training_loss() will give the training_loss (negative elbo) for a given batch
                elbo = -training_loss().numpy()
                # check if new elbo estimate is larger than previous
                if (elbo > max_elbo) & (early_stop):
                    max_elbo = elbo
                    max_count = 0
                    # TODO: store parameters
                    # - will this copy by reference? yes(?)

                    best_step = step
                    if save_best:
                        params = copy.deepcopy(parameter_dict(model))
                    # else:
                    #     params = parameter_dict(model).copy()
                else:
                    max_count += 1
                    if (max_count > persistance) & (early_stop):
                        print("objective did not improve stopping")
                        stopped_early = True
                        break

                logf.append(elbo)
        return logf, params, best_step, stopped_early


    def _svgp_optimise(self,
                       use_minibatch=True,
                       gamma=0.5,
                       learning_rate=0.05,
                       trainable_inducing_variable=False,
                       minibatch_size=100,
                       maxiter=20000,
                       log_freq=10,
                       persistence=100,
                       early_stop=True,
                       save_best=False):
        """"""
        # TODO: tidy up _svgp_optimise

        assert self.engine == "GPflow_svgp", f"engine={self.engine}, expected 'GPflow_svgp'"
        # elbo = tf.function(self.model.elbo)

        # data = (X, Y)
        data = (self.x, self.y)

        # tensor_data = tuple(map(tf.convert_to_tensor, data))

        # %timeit elbo(tensor_data)

        # minibatch_size = 100

        # We turn off training for inducing point locations
        # gpflow.set_trainable(self.model.inducing_variable, train_inducing_points)

        # train_dataset - copied from
        N = len(self.x)

        if use_minibatch:
            # TODO: review the impacts of repeat and shuffle on tf.Dataset
            # train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(N)
            autotune = tf.data.experimental.AUTOTUNE
            train_dataset = (
                tf.data.Dataset.from_tensor_slices(data)
                .prefetch(autotune)
                .repeat()
                .shuffle(N)
                .batch(minibatch_size)
            )
            train_iter = iter(train_dataset)
            loss_fn = self.model.training_loss_closure(train_iter, compile=True)

        else:
            loss_fn = self.model.training_loss_closure(data)

        # make q_mu and q_sqrt non training to adam
        gpflow.utilities.set_trainable(self.model.q_mu, False)
        gpflow.utilities.set_trainable(self.model.q_sqrt, False)

        # select the variational parameters for natural gradients
        variational_vars = [(self.model.q_mu, self.model.q_sqrt)]
        natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=gamma)

        # make the inducing variable trainable ?
        gpflow.set_trainable(self.model.inducing_variable, trainable_inducing_variable)

        # parameters for adam to train
        adam_vars = self.model.trainable_variables
        adam_opt = tf.optimizers.Adam(learning_rate)

        # each optimisation step will update variational and then model (GP?) parameters
        @tf.function
        def optimisation_step():
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)

        # --
        # iterate the variables
        # --

        # TODO: in _svgp_optimise variable iteration needs cleaning up
        t0 = time.time()
        # initialise the maximum elbo
        max_elbo = -np.inf
        # best_step = 0
        stopped_early = False
        # NOTE: need to use copy.deepcopy() - this could slow things down
        params = parameter_dict(self.model)

        logf = []

        for step in range(maxiter):
            optimisation_step()
            if step % log_freq == 0:
                # training_loss() will give the training_loss (negative elbo) for a given batch
                elbo = -loss_fn().numpy()
                if self.verbose > 2:
                    print(f"step: {step},  elbo: {elbo:.2f}")
                # check if new elbo estimate is larger than previous
                if (elbo > max_elbo) & (early_stop):
                    max_elbo = elbo
                    max_count = 0
                    # TODO: store parameters
                    # - will this copy by reference? yes(?)
                    # best_step = step
                    if save_best:
                        params = copy.deepcopy(parameter_dict(self.model))
                    # else:
                    #     params = parameter_dict(model).copy()
                else:
                    max_count += log_freq
                    if (max_count >= persistence) & (early_stop):
                        print("objective did not improve stopping")
                        stopped_early = True
                        break

                logf.append(elbo)
            # return logf, params, best_step, stopped_early

        t1 = time.time()
        if self.verbose > 3:
            print(f"opt run time for svgp: {t1-t0:.2f}s")
        # run adam optimizer
        # TODO: add print statements (for a given verbose level)
        # logf, params, best_step, stopped_early = self._svgp_run_adam(model=self.model,
        #                                                              iterations=maxiter,
        #                                                              train_dataset=train_dataset,
        #                                                              minibatch_size=minibatch_size,
        #                                                              early_stop=early_stop,
        #                                                              persistance=persistance,
        #                                                              save_best=save_best)

        # load parameters from "best" epoch (possibly from mini batch, so might not reflect full data)
        if save_best:
            if self.verbose > 3:
                print("loading 'best' parameters found")
            gpflow.utilities.multiple_assign(self.model, params)

        # consider a successful optimisation of stopped early (with early stop being on)
        success = stopped_early if stopped_early else np.nan
        return {"success": success, "elbo": logf}



    def get_neighbours_of_grid_loc(self, grid_loc,
                                   coarse_grid_spacing=1,
                                   flatten=True):
        """get x,y location about some grid location"""

        # if not using_raw data select values
        if not self.obs_date['raw_data']:
            gl0 = grid_loc[0] + np.arange(-coarse_grid_spacing, coarse_grid_spacing + 1)
            gl1 = grid_loc[1] + np.arange(-coarse_grid_spacing, coarse_grid_spacing + 1)

            # trim to be in grid range
            gl0 = gl0[(gl0 >= 0) & (gl0 < self.aux['y'].vals.shape[1])]
            gl1 = gl1[(gl1 >= 0) & (gl1 < self.aux['x'].vals.shape[1])]

            gl0, gl1 = np.meshgrid(gl0, gl1)

            # location to predict on
            x_pred = self.aux['x'].vals[gl0, gl1]
            y_pred = self.aux['y'].vals[gl0, gl1]
            t_pred = np.zeros(x_pred.shape)

        else:
            # get the x,y valyes
            x, y = self.aux['x'].vals[grid_loc], self.aux['y'].vals[grid_loc]

            gr = self.grid_res
            # if grid_res is a string - i.e. expect 50km
            if isinstance(gr, str):
                gr = int(re.sub("\D", "", gr))
            # half of the interval - assuming x,y is at center of square
            half_ival = (gr * 1000) / 2

            hdims = self.obs_date['held_out'].dims
            b = (hdims['x'] >= (x - half_ival)) & \
                (hdims['x'] < (x + half_ival)) & \
                (hdims['y'] >= (y - half_ival)) & \
                (hdims['y'] < (y + half_ival))

            x_pred = hdims['x'][b]
            y_pred = hdims['y'][b]
            t_pred = hdims['t'][b]

            # add prediction at center
            x_pred = np.concatenate([x_pred, np.array([x])])
            y_pred = np.concatenate([y_pred, np.array([y])])
            t_pred = np.concatenate([t_pred, np.array([0])])

            gl0, gl1 = np.full(x_pred.shape, np.nan), np.full(x_pred.shape, np.nan)

        out = [x_pred, y_pred, t_pred, gl0, gl1]

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
            if self.verbose > 1:
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
            xs = xs[None, :]
        assert xs.shape[1] == self.x.shape[1], \
            f"dimension of test point(s): {xs.shape} is not aligned to x/input data: {self.x.shape}"

        # scale input values
        xs *= self.scale_inputs

        out = {}
        if self.engine in ["GPflow", "GPflow_svgp"]:
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
        if isinstance(sat_names, str):
            sat_names = [sat_names]

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
        sie = self.sie.vals  # ['data']
        sie_dates = self.sie.dims['date']  # ['dims']['date']

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

    @staticmethod
    def hyper_params_for_date(date, prev_res=None, dims=None, length_scale_name=None):
        # TODO: change hyper_params_for_date from a static method
        assert date is not None
        out = {}
        select_dims = {
            "date": date
        }
        if prev_res is not None:

            # get the length scale - start with
            for ls in [k for k in prev_res.keys() if re.search('^ls', k)]:
                out[ls] = prev_res[ls].subset(select_dims=select_dims)

            for _ in ['kernel_variance', "likelihood_variance"]:
                out[_] = prev_res[_].subset(select_dims=select_dims)
        else:
            # TODO: here should allow for setting of hyper parameters
            assert isinstance(dims, dict), "res is None, dims needs to be a dict"
            assert isinstance(length_scale_name, (list, tuple, np.array))

            if 'date' not in dims:
                dims['date'] = np.array([date])

            # ls_keys = [k for k in length_scale_name if re.search('^ls', k)]
            ls_keys = [f"ls_{k}" for k in length_scale_name]
            var_keys = ['kernel_variance', "likelihood_variance"]
            for _ in ls_keys + var_keys:
                # NOTE: this could be inefficient if dates in dim is big
                out[_] = DataDict.full(dims=dims, fill_val=1.0, name=_).subset(select_dims)

        return out

    @staticmethod
    def hyper_params_for_date_and_grid_loc(res, date, grid_loc,
                                           ls_order=None):
        assert len(grid_loc) == 2, "grid_loc should be a tuple of length 2"

        # TODO: allow for hyper parameters to be missing, if missing provide None
        out = {}
        select_dims = {
            "date": date,
            "grid_loc_0": grid_loc[0],
            "grid_loc_1": grid_loc[1]
        }

        # get the length scale - start with
        ls_vals = {}
        for ls in [k for k in res.keys() if re.search('^ls', k)]:
            try:
                ls_vals[ls] = res[ls].subset(select_dims=select_dims)
            except Exception as e:
                print(f"Error: {e}")
                _ = 1
                assert False
        ls_order = list(ls_vals.keys()) if ls_order is None else ls_order
        out['length_scales'] = [np.squeeze(ls_vals[ls].vals) for ls in ls_order]
        out['length_scales'], = to_array(out['length_scales'])

        for _ in ['kernel_variance', "likelihood_variance"]:
            tmp = res[_].subset(select_dims=select_dims)
            out[_] = np.squeeze(tmp.vals)

        # TODO: handle the variance floor differently
        if out['kernel_variance'] <= 1e-6:
            print("kernel variance too low")
            out['kernel_variance'] = 1.01e-6
        if out['likelihood_variance'] <= 1e-6:
            print("likelihood variance too low")
            out['likelihood_variance'] = 1.01e-6

        return out

    def post_process(self,
                     date,
                     prev_results_file=None,
                     prev_results_dir=None,
                     clip_and_smooth=False,
                     smooth_method="kernel",
                     vmin_map=None,
                     vmax_map=None,
                     std=None,
                     grid_res=None,
                     big_grid_size=360,
                     prev_file_suffix=None):

        if prev_file_suffix is not None:
            warnings.warn(f"prev_file_suffix is not None, it's: {prev_file_suffix}, however it's not used by the post_process method")

        # TODO: add option to provide previous result directly to post_process - instead of reading in from file
        assert grid_res is not None, "grid_res needs to be specified"
        assert not isinstance(prev_results_file, dict), "prev_result type: dict not yet implemented"

        assert isinstance(prev_results_file, (str, type(None)))

        if prev_results_file is None:
            if self.verbose:
                print("in post_process: prev_result is None, will use naive hyper parameters")
            hp_date = self.hyper_params_for_date(date=date, prev_res=None,
                                                 dims=self.aux['x'].dims.copy(),
                                                 length_scale_name=self.length_scale_name)

        elif isinstance(prev_results_file, dict):
            assert False, "prev_result type: dict not yet implemented"
            # return hp_date
        elif isinstance(prev_results_file, str):

            assert prev_results_dir is not None
            assert os.path.exists(prev_results_dir), f"prev_results_dir:\n{prev_results_dir}\ndoes not exist"

            # read in previous results
            prev_res = self.read_results(results_dir=prev_results_dir,
                                         file=prev_results_file,
                                         grid_res_loc=grid_res,
                                         grid_size=big_grid_size,
                                         unflatten=True,
                                         dates=[date])

            # ---
            # get the hyper parameters for the date
            # ---

            # TODO: consider storing hp_date as attributes
            # - store hp_date as attribute
            hp_date = self.hyper_params_for_date(date=date, prev_res=prev_res)

            # ---
            # apply clipping / smoothing
            # ---

            if clip_and_smooth:
                # std = 50 / grid_res
                assert std is not None, f"std in post process is None, please specify"
                # --
                # sie mask
                # --
                # use SIE to create a mask when clipping and smoothing hyper parameters
                sie_mask = self.sie.subset(select_dims={'date': date})
                sie_mask = np.isnan(sie_mask.vals)
                sie_mask = np.squeeze(sie_mask)

                # - be in 'post_process' config
                for k in hp_date.keys():
                    hp_date[k] = hp_date[k].clip_smooth_by_date(smooth_method=smooth_method,
                                                                nan_mask=sie_mask,
                                                                vmin=vmin_map[k] if isinstance(vmin_map,
                                                                                               dict) else None,
                                                                vmax=vmax_map[k] if isinstance(vmax_map,
                                                                                               dict) else None,
                                                                std=std)
        return hp_date

    @staticmethod
    def make_temp_dir(incl_rad,
                      days_ahead,
                      days_behind,
                      grid_res,
                      season,
                      coarse_grid_spacing,
                      hold_out,
                      bound_length_scales,
                      prior_mean_method):

        if hold_out is None:
            hold_out_str = ""
        elif isinstance(hold_out, str):
            hold_out_str = re.sub("_", "", hold_out)
        else:
            hold_out_str = '|'.join([re.sub("_", "", ho) for ho in hold_out])
        # remove underscores from prior mean - just to include in output directory
        priomean_str = re.sub('_', "", prior_mean_method)

        tmp_dir = f"radius{incl_rad}_daysahead{days_ahead}_daysbehind{days_behind}_" \
                  f"gridres{grid_res}_season{season}_coarsegrid{coarse_grid_spacing}_" \
                  f"holdout{hold_out_str}_boundls{bound_length_scales}_meanMeth{priomean_str}"

        return tmp_dir

    def run(self,
            date,
            output_dir,
            days_ahead=4,
            days_behind=4,
            incl_rad=300,
            grid_res=25,
            coarse_grid_spacing=1,
            min_inputs=10,
            min_sie=0.15,
            engine="GPflow",
            kernel="Matern32",
            season='2018-2019',
            prior_mean_method="fyi_average",
            optimise=True,
            load_params=False,
            hold_out=None,
            scale_inputs=False,
            scale_outputs=False,
            append_to_file=True,
            overwrite=True,
            pred_on_hold_out=True,
            bound_length_scales=True,
            mean_function=None,
            file_suffix="",
            post_process=None,
            print_every=100,
            inducing_point_params=None,
            optimise_params=None):
        """
        wrapper function to run optimal interpolation of sea ice freeboard for a given date
        """

        # TODO: move min_obs_for_svgp into inducing_point_params
        # TODO: allow reading from previous results - for intialisation

        # res = sifb.read_results(results_dir, file="results.csv", grid_res_loc=grid_res, grid_size=big_grid_size,
        #                         unflatten=True)

        result_file = f"results{file_suffix}.csv"
        prediction_file = f"prediction{file_suffix}.csv"
        config_file = f"input_config{file_suffix}.json"
        skipped_file = f"skipped{file_suffix}.csv"
        param_file = f"params{file_suffix}"

        if post_process is None:
            post_process = {}

        if inducing_point_params is None:
            inducing_point_params = {}

        if optimise_params is None:
            optimise_params = {}

        # TODO: refactor getting min_obs_for_svgp from inducing_point_params - review how it's used down stream
        min_obs_for_svgp = inducing_point_params.get("min_obs_for_svgp", 1000)
        if "min_obs_for_svgp" in inducing_point_params:
            inducing_point_params.pop("min_obs_for_svgp")

        # ----
        # get the input parameters
        # ---

        # taken from answers:
        # https://stackoverflow.com/questions/218616/how-to-get-method-parameter-names
        config = {}
        locs = locals()
        for k in range(self.run.__code__.co_argcount):
            var = self.run.__code__.co_varnames[k]
            config[var] = locs[var]
        # config['kwargs'] = locs['kwargs']

        # ---
        # modify / interpret parameters
        # ---

        # HARDCODED: the amount of scaling that is applied
        # TODO: allow this to be specified by user - do the default if scale_inputs=True
        if isinstance(scale_inputs, bool):
            scale_inputs = [1 / (grid_res * 1000), 1 / (grid_res * 1000), 1.0] if scale_inputs else [1.0, 1.0, 1.0]
            if self.verbose:
                print(f"using these values to scale_inputs: {scale_inputs}")

        if isinstance(scale_outputs, bool):
            scale_outputs = 100. if scale_outputs else 1.
            if self.verbose:
                print(f"using these values to scale_outputs: {scale_outputs}")

        assert len(scale_inputs) == 3, "scale_inputs expected to be length 3"

        # HARDCODED: length scale bounds
        # TODO: allow values to be specified by user
        if bound_length_scales:
            # NOTE: input scaling will happen in model build (for engine: GPflow)
            ls_lb = np.zeros(len(scale_inputs))
            ls_ub = np.array([(2 * incl_rad * 1000),
                              (2 * incl_rad * 1000),
                              (days_behind + days_ahead + 1)])

            if self.verbose:
                print(f"length scale:\nlower bounds: {ls_lb}\nupper bounds: {ls_ub}")

        else:
            ls_lb, ls_ub = None, None

        # -
        # results dir
        # -

        # TODO: tidy up how the sub-directory gets added to output dir
        # results_dir = output_dir

        # subdirectory in results dir, with name containing (some) run parameters
        # if hold_out is None:
        #     hold_out_str = ""
        # elif isinstance(hold_out, str):
        #     hold_out_str = re.sub("_", "", hold_out)
        # else:
        #     hold_out_str = '|'.join([re.sub("_", "", ho) for ho in hold_out])
        # # remove underscores from prior mean - just to include in output directory
        # priomean_str = re.sub('_', "", prior_mean_method)
        # tmp_dir = f"radius{incl_rad}_daysahead{days_ahead}_daysbehind{days_behind}_gridres{grid_res}_season{season}_coarsegrid{coarse_grid_spacing}_holdout{hold_out_str}_boundls{bound_length_scales}_meanMeth{priomean_str}"
        tmp_dir = self.make_temp_dir(incl_rad,
                                     days_ahead,
                                     days_behind,
                                     grid_res,
                                     season,
                                     coarse_grid_spacing,
                                     hold_out,
                                     bound_length_scales,
                                     prior_mean_method)

        print(f"will write results to subdir of output_dir:\n {tmp_dir}")
        output_dir = os.path.join(output_dir, tmp_dir)
        os.makedirs(output_dir, exist_ok=True)

        # ---
        # run info
        # ---

        # run time info
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        run_info = {
            "run_datetime": now,
            "raw_data": self.raw_obs is None
        }

        # add git_info
        try:
            run_info['git_info'] = get_git_information()
        except subprocess.CalledProcessError:
            print("issue getting git_info, check current working dir")
            pass

        config["run_info"] = run_info

        # ---
        # write config to file - will end up doing this for each date
        # ---

        config.pop('self')
        with open(os.path.join(output_dir, config_file), "w") as f:
            json.dump(config, f, indent=4)

        # record time to run
        t_total0 = time.time()

        # ----
        # for each date: select data used to build GP
        # ----

        all_res = []
        all_preds = []

        # for date in dates:
        print(f"date: {date}")
        # --
        # date directory and file name
        # --

        date_dir = os.path.join(output_dir, date)
        os.makedirs(date_dir, exist_ok=True)

        # results will be written to file

        res_file = os.path.join(date_dir, result_file)
        pred_file = os.path.join(date_dir, prediction_file)
        # bad results will be written to
        skip_file = os.path.join(date_dir, skipped_file)

        # dictionary of full path to files will be returned by run method
        output_files = {
            "results": res_file,
            "predictions": pred_file,
            "skipped": skip_file
        }

        # ---
        # move files to archive, if they already exist
        # ---

        # TODO: allow for appending to existing data
        #  - allow to read in existing data then check if grid_loc already exists

        prior_rdf = pd.DataFrame(columns=["grid_loc_0", "grid_loc_1"])
        if overwrite:
            param_files = [param_file + _ for _ in ['.bak', '.dir', '.dat', '.pkl']]
            move_to_archive(top_dir=date_dir,
                            file_names=[config_file,
                                        result_file,
                                        prediction_file,
                                        skipped_file] + param_files,
                            suffix=f"_{now}",
                            verbose=True)

        else:
            try:
                # TODO: allow this
                if optimise:
                    prior_rdf = pd.read_csv(os.path.join(date_dir, result_file))
                    prior_rdf = prior_rdf[["grid_loc_0", "grid_loc_1"]]
                else:
                    print(f"overwrite={overwrite} but optimise={optimise}, "
                          f"will load params (if available) and generated predictions")
            except FileNotFoundError as e:
                print(f"previous result_file: {result_file}")

        # ---
        # write config to file - will end up doing this for each date
        # ---

        with open(os.path.join(date_dir, config_file), "w") as f:
            json.dump(config, f, indent=4)

        # -------
        # hyper parameters for date (optionally post-processed if read from else where)
        # -------

        post_process["prev_results_dir"] = post_process.get("prev_results_dir", output_dir)
        post_process["prev_results_file"] = post_process.get("prev_results_file", None)

        hp_date = self.post_process(date=date,
                                    grid_res=grid_res,
                                    std=50 / grid_res,
                                    **post_process)

        # --
        # select data for given date
        # --

        # TODO: allow prior mean method to be specified differently
        self.select_data_for_given_date(date=date,
                                        days_ahead=days_ahead,
                                        days_behind=days_behind,
                                        hold_out=hold_out,
                                        prior_mean_method=prior_mean_method,
                                        min_sie=None)

        # ----
        # locations to calculate GP on
        # ----

        gp_locs = self.select_gp_locations(date=date,
                                           min_sie=min_sie,
                                           coarse_grid_spacing=coarse_grid_spacing,
                                           sat_names=hold_out if pred_on_hold_out else None)
        select_loc = np.where(gp_locs)

        # ---
        # for each location to optimise GP
        # ---

        num_loc = gp_locs.sum()
        print(f"will calculate GPs at: {num_loc} locations")

        # TODO: review why self.verbose is set to 0 here
        # self.verbose = 0

        # ---
        # model parameters - load previously generated?
        # ---

        # store parameters (from GPflow) in dict
        param_dict = {}

        # TODO: tidy up the reading in of previous model parameters - make this into a method?
        # if the engine is not PurePython
        if engine != "PurePython":
            # load pre-existing model parameters, if they exist
            prev_data_dir = post_process.get('prev_results_dir', None)
            if prev_data_dir is not None:
                prev_data_dir = os.path.join(prev_data_dir, date)
                assert os.path.exists(prev_data_dir), f"prev_data_dir:\n{prev_data_dir}\nwas provide but does not exist"
                prev_file_suffix = post_process.get("prev_file_suffix", "")
                prev_param_file = os.path.join(prev_data_dir, f"params{prev_file_suffix}")

                with shelve.open(prev_param_file) as sdb:
                    for k in sdb.keys():
                        if k not in param_dict:
                            # NOTE: the keys in shelve objects are strings, expect them to be able to be converted to tuples
                            param_dict[make_tuple(k)] = sdb[k]

                print(f"loaded parameters for: {len(param_dict)} GP models")

        for i in range(num_loc):

            if (i % print_every) == 0:
                print("*" * 75)
                print(f"{i + 1}/{num_loc + 1}")

            t0 = time.time()
            # ---
            # select location
            # ---

            grid_loc = select_loc[0][i], select_loc[1][i]

            # check
            # could this be slow?
            exists = (prior_rdf['grid_loc_0'] == grid_loc[0]) & (prior_rdf['grid_loc_1'] == grid_loc[1])
            if exists.any():
                if self.verbose > 2:
                    print(f"grid_loc: {grid_loc} already has results, skipping")
                continue

            x_ = self.aux['x'].vals[grid_loc]
            y_ = self.aux['y'].vals[grid_loc]

            xy_loc = (x_, y_)

            # alternatively could use
            # x_ = self.obs.dims['x'][grid_loc[1]]
            # y_ = self.obs.dims['y'][grid_loc[0]]

            # select inputs for a given location
            inputs, outputs = self.select_input_output_from_obs_date(x=x_,
                                                                     y=y_,
                                                                     incl_rad=incl_rad)

            if self.verbose > 2:
                print(f"number of inputs/outputs: {len(outputs)}")

            # TODO: move this into a method
            # too few inputs?
            if len(inputs) < min_inputs:
                # print("too few inputs")
                tmp = pd.DataFrame({"grid_loc_0": grid_loc[0],
                                    "grid_loc_1": grid_loc[1],
                                    "reason": f"had only {len(inputs)} inputs"},
                                   index=[i])

                tmp.to_csv(skip_file, mode='a',
                           header=not os.path.exists(skip_file),
                           index=False)
                continue

            # HACK: prior_mean_method == "demean_outputs"
            # - here de-mean the outputs, the mean subtracted will depend on the points in radius
            if prior_mean_method == "demean_outputs":
                output_mean = np.mean(outputs)
                self.mean.vals[grid_loc[0], grid_loc[1]] += output_mean
                outputs -= output_mean

            # ---
            # get hyper parameters - for the given date and location
            # ---

            # hps = hyper_params_for_date_and_grid_loc(res, date, grid_loc,
            #                                          ls_order=ls_order)

            # get the length scale order
            ls_order = [f"ls_{i}" for i in self.length_scale_name]
            hps = self.hyper_params_for_date_and_grid_loc(hp_date, date, grid_loc,
                                                          ls_order=ls_order)

            # HACK: just skipping points where kernel_variance is missing
            #  - this would be from previously loaded data
            if np.isnan(hps['kernel_variance']):
                if self.verbose:
                    print(f'kernel_variance is nan, skipping this (grid) location: {grid_loc}')
                continue

            # ---
            # build a GPR model for data
            # ---

            self.build_gpr(inputs=inputs,
                           outputs=outputs,
                           scale_inputs=scale_inputs,
                           scale_outputs=scale_outputs,
                           length_scales=hps['length_scales'],
                           kernel_var=hps['kernel_variance'],
                           likeli_var=hps['likelihood_variance'],
                           length_scale_lb=ls_lb,
                           length_scale_ub=ls_ub,
                           engine=engine,
                           kernel=kernel,
                           mean_function=mean_function,
                           min_obs_for_svgp=min_obs_for_svgp,
                           **inducing_point_params)

            # ----
            # load model parameters
            # ----

            if (self.engine != "PurePython") & load_params:

                if xy_loc in param_dict:
                    if self.verbose > 1:
                        print("loading previous model parameters")
                    # NOTE: if min_obs_for_svgp is different from last time it was run
                    # - this could lead to self.model being a different type
                    # - and this will cause an issue
                    multiple_assign(self.model, param_dict[xy_loc])

            # ---
            # get the hyper parameters
            # ---

            # hps = self.get_hyperparameters(scale_hyperparams=False)

            # ---
            # optimise model
            # ---

            if optimise:
                t0_opt = time.time()
                opt_hyp = self.optimise(scale_hyperparams=False, **optimise_params)
                t1_opt = time.time()
                opt_runtime = t1_opt - t0_opt
                if self.verbose > 2:
                    print(f"opt_runtime: {opt_runtime:.2f}s")
            else:
                if self.verbose > 2:
                    print("not optimising hyper parameters")
                opt_hyp = self.get_hyperparameters(scale_hyperparams=False)
                opt_hyp["marginal_loglikelihood"] = self.get_marginal_log_likelihood()
                opt_hyp["optimise_success"] = np.nan
                opt_runtime = np.nan

            t1 = time.time()

            # ---
            # get the marginal log likelihood
            # --

            # mll = self.get_marginal_log_likelihood()

            # ---
            # make predictions
            # ---

            # TODO: predict in region around center point
            x_pred, y_pred, t_pred, gl0, gl1 = self.get_neighbours_of_grid_loc(grid_loc,
                                                                       coarse_grid_spacing=coarse_grid_spacing)

            preds = self.predict_freeboard(x=x_pred,
                                           y=y_pred,
                                           t=t_pred)

            # ----
            # store model parameters
            # ----

            # TODO: allow for parameter extraction from PurePython, similar to GPflow (low priority)
            if self.engine != "PurePython":
                with shelve.open(os.path.join(date_dir, param_file), writeback=True) as sdb:
                    sdb[repr(xy_loc)] = parameter_dict(self.model)

            # ----
            # store values in DataFrame
            # ---

            # TODO: storing values in dict/ DAtaFrame should be tidied up
            t2 = time.time()
            preds['grid_loc_0'] = grid_loc[0]
            preds['grid_loc_1'] = grid_loc[1]
            preds["proj_loc_0"] = gl0
            preds["proj_loc_1"] = gl1
            # TODO: this needs to be more robust to handle different mean priors
            # - using the mean of the single GP just calculated
            preds['fyi_mean'] = self.mean.vals[grid_loc[0], grid_loc[1]]
            # TODO: getting mean is quite hard coded -
            preds['mean'] = self.mean.vals[grid_loc[0], grid_loc[1]] + opt_hyp.get("mean_func_c", 0)

            preds['date'] = date

            # the split the test values per dimension
            xs = preds.pop('xs')
            for i in range(xs.shape[1]):
                preds[f'xs_{self.length_scale_name[i]}'] = xs[:, i]

            preds['run_time'] = t2 - t1

            # ----
            # store results (parameters) and predictions
            # ----

            ln, lt = EASE2toWGS84_New(x_, y_)

            # store predictions at 'center' location in results
            # center_loc_bool = (preds['proj_loc_0'] == preds['grid_loc_0']) & (
            #             preds['proj_loc_1'] == preds['grid_loc_1'])

            # get the center of grid prediction
            center_loc_bool = (xs[:, 0] == (x_ * self.scale_inputs[0])) & \
                              (xs[:, 1] == (y_ * self.scale_inputs[1])) & \
                              (xs[:, 2] == (0 * self.scale_inputs[2]))

            center_pred = {_: preds[_][center_loc_bool] if isinstance(preds[_], np.ndarray) else np.array([preds[_]])
                           for _ in ["f*", "f*_var", "y_var", "mean", "fyi_mean"]}
            # center_pred['mean'] = preds['mean']
            # center_pred['fyi_mean'] = preds['fyi_mean']

            res = {
                "date": date,
                "x_loc": x_,
                "y_loc": y_,
                "lon": ln,
                "lat": lt,
                "grid_loc_0": grid_loc[0],
                "grid_loc_1": grid_loc[1],
                "num_inputs": len(inputs),
                "output_mean": np.mean(outputs),
                "output_std": np.std(outputs),
                "engine": self.engine,
                "gpu_name": self.gpu_name,
                **center_pred,
                **opt_hyp,
                "run_time": t1 - t0,
                "opt_runtime": opt_runtime,
                **{f"scale_{self.length_scale_name[i]}": si
                   for i, si in enumerate(self.scale_inputs)},
                "scale_output": self.scale_outputs
            }

            # store in dataframe - for easy writing / appending to file

            # TEMP DEBUGGING FOR COLAB
            try:
                rdf = pd.DataFrame(res, index=[i])
            except TypeError as e:
                print("error in putting res into data Frame")
                print(e)
                print("res:")
                print(res)
                print(f"i: {i}")

                assert False

            pdf = pd.DataFrame(preds)

            if append_to_file:
                # append results to file
                rdf.to_csv(res_file, mode="a", header=not os.path.exists(res_file),
                           index=False)
                pdf.to_csv(pred_file, mode="a", header=not os.path.exists(pred_file),
                           index=False)

            all_res.append(rdf)
            all_preds.append(pdf)

        try:
            all_res = pd.concat(all_res)
            all_preds = pd.concat(all_preds)
        except ValueError:
            if self.verbose > 1:
                print("no results to concatenate, will return None, None")
            all_res = None
            all_preds = None
        # --
        # total run time
        # --

        t_total1 = time.time()
        print(f"total run time: {t_total1 - t_total0:.2f}")

        with open(os.path.join(output_dir, "total_runtime.txt"), "+w") as f:
            f.write(f"runtime: {t_total1 - t_total0:.2f} seconds")

        return all_res, all_preds, output_files

    def get_results_from_dir(self,
                             res_dir,
                             dates=None,
                             results_file="results.csv",
                             predictions_file="prediction.csv",
                             file_suffix="",
                             big_grid_size=360,
                             results_data_cols=None,
                             preds_data_cols=None
                             ):
        """a wrapper for read_results, getting both data from results_file and predictions
        outputs combined with input_config"""
        # TODO: remove commented sections of code below, and un used inputs
        # get the config file to determine how it was created
        with open(os.path.join(res_dir, f"input_config{file_suffix}.json"), "r") as f:
            config = json.load(f)

        # --
        # extract parameters
        # --

        # TODO: in get_results_from_dir can more parameters be fetched from config?
        grid_res = config['grid_res']

        # --
        # read results - hyper parameters values, log likelihood
        # --

        if results_file is not None:
            res = self.read_results(res_dir,
                                    file=results_file,
                                    grid_res_loc=grid_res,
                                    grid_size=big_grid_size,
                                    unflatten=True,
                                    dates=dates, file_suffix=file_suffix,
                                    data_cols=results_data_cols)
        else:
            res = {}

        # --
        # read predictions - f*, f*_var, etc
        # --

        # TODO: at some point this will not be needed because  f*, f*_var, etc
        #  is now stored in 'results' (i.e. with hyper parameter)
        if predictions_file is not None:
            pre = self.read_results(res_dir,
                                    file=predictions_file,
                                    grid_res_loc=grid_res,
                                    grid_size=big_grid_size,
                                    unflatten=True,
                                    dates=dates,
                                    file_suffix=file_suffix,
                                    data_cols=preds_data_cols)
        else:
            pre = {}

        # ---
        # combine dicts
        # ---

        #
        common_keys = np.intersect1d(list(res.keys()), list(pre.keys()))
        for ck in common_keys:
            pre[ck + "_pred"] = pre.pop(ck)

        out = {**res, **pre}

        out['input_config'] = config

        # out['lon_grid'] = sifb.aux['lon']
        # out['lat_grid'] = sifb.aux['lat']

        return out

    def cross_validation_results(self,
                                 prev_results=None,
                                 prev_results_dir=None,
                                 hold_out=None,
                                 add_mean=True,
                                 load_data=False,
                                 **kwargs):
        """cross validation results
        - kwargs are for get_results_from_dir()"""

        if prev_results is None:
            assert prev_results_dir is not None
            prev_results = self.get_results_from_dir(
                res_dir=prev_results_dir,
                **kwargs)

        assert isinstance(prev_results, dict)
        assert "f*" in prev_results
        assert "y_var" in prev_results

        config = prev_results['input_config']

        if hold_out is None:
            hold_out = config['hold_out']

        # load data
        if load_data:
            grid_res = config['grid_res']
            season = config['season']
            data_dir = config['data_dir']

            if data_dir == "package":
                data_dir = get_data_path()

            assert os.path.exists(data_dir)

            self.load_data(aux_data_dir=os.path.join(data_dir, "aux"),
                           sat_data_dir=os.path.join(data_dir, "CS2S3_CPOM"),
                           grid_res=grid_res,
                           season=season)

        assert self.obs is not None, "obs attribute is None, please make sure correct observations are loaded"

        # ---
        # evaluate predictions
        # ---

        # store results in list
        z_list = []
        dif_list = []
        ystd_list = []
        stat_list = []
        llz_list = []
        ll_list = []

        for date in prev_results['f*'].dims['date']:

            print(date)

            # select prediction data
            select_dims = {"date": date}
            fs = prev_results['f*'].subset(select_dims)
            if add_mean:
                assert 'mean' in prev_results
                fs = fs + prev_results['mean'].subset(select_dims)

            y_var = prev_results['y_var'].subset(select_dims)

            # date_dims = fs.dims

            # extract numpy array, squeeze date dim
            fs = np.squeeze(fs.vals)
            y_std = np.sqrt(np.squeeze(y_var.vals))

            # evaluate each held out satellite data
            z_tmp = []
            dif_tmp = []
            ystd_tmp = []
            llz_tmp = []
            ll_tmp = []
            for ho in hold_out:
                # select the data for a held out satellite data
                hold_dims = {"date": date, "sat": ho}
                obs = self.obs.subset(hold_dims)
                obs = np.squeeze(obs.vals)

                # get the difference
                dif = (obs - fs)
                # normalised

                z = dif / y_std
                # test statistic
                # TODO: add more test statistics for cross val?
                z_non_nan = ~np.isnan(z)
                c = shapiro(z[z_non_nan])
                # the log likelihood of the z measurements
                # - z assumed to standard normal
                ll_z = norm.logpdf(z[z_non_nan])
                # log likelihood with variable sigma (variance)
                ll = norm.logpdf(dif[z_non_nan],
                                 loc=0,
                                 scale=y_std[z_non_nan])

                stats = {
                    "date": date,
                    "sat": ho,
                    "shapiro_statistic": c.statistic,
                    "shaprio_pvalue": c.pvalue,
                    "log_likelihood": ll.sum(),
                    "log_likelihood_z": ll_z.sum(),
                    "num_obs": z_non_nan.sum()
                }
                stat_list.append(stats)

                # store the differences as DataDicts, then in list
                _ = DataDict(vals=z, name=ho, default_dim_name="grid_loc_")
                z_tmp.append(_)
                _ = DataDict(vals=dif, name=ho, default_dim_name="grid_loc_")
                dif_tmp.append(_)
                _ = DataDict(vals=y_std, name=ho, default_dim_name="grid_loc_")
                ystd_tmp.append(_)
                # NOTE: here the using idx because arrays are 1-d, instead of the 2-d above
                _ = DataDict(vals=ll_z, name=ho, default_dim_name="idx")
                llz_tmp.append(_)
                _ = DataDict(vals=ll, name=ho, default_dim_name="idx")
                ll_tmp.append(_)

            # concatenate across the different satellites in hold_out (could just be 1)
            _ = DataDict.concatenate(*z_tmp, dim_name='sat', name=date, verbose=False)
            z_list.append(_)
            _ = DataDict.concatenate(*dif_tmp, dim_name='sat', name=date, verbose=False)
            dif_list.append(_)
            _ = DataDict.concatenate(*ystd_tmp, dim_name='sat', name=date, verbose=False)
            ystd_list.append(_)
            _ = DataDict.concatenate(*llz_tmp, dim_name='sat', name=date, verbose=False)
            llz_list.append(_)
            _ = DataDict.concatenate(*ll_tmp, dim_name='sat', name=date, verbose=False)
            ll_list.append(_)


        # concatenate across dates
        zdd = DataDict.concatenate(*z_list, dim_name='date', name='norm_diff', verbose=False)
        difdd = DataDict.concatenate(*dif_list, dim_name='date', name="diff", verbose=False)
        ystddd = DataDict.concatenate(*ystd_list, dim_name='date', name="y_std", verbose=False)
        llzdd = DataDict.concatenate(*llz_list, dim_name='date', name="y_std", verbose=False)
        lldd = DataDict.concatenate(*ll_list, dim_name='date', name="y_std", verbose=False)


        # return a dictionary of results
        out = {
            "stats": pd.DataFrame(stat_list),
            "diff": difdd,
            "z": zdd,
            "y_std": ystddd,
            "ll_z": llzdd,
            "ll": lldd
        }
        return out


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
