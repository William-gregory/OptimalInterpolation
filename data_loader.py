# DataLoader Class

import numpy as np
import pandas as pd
import os
import json
import re
import pickle
import warnings

from functools import reduce

from scipy.interpolate import interp2d, griddata

from OptimalInterpolation.data_dict import DataDict
from OptimalInterpolation.utils import readFB, split_data2, grid_proj, WGS84toEASE2_New, EASE2toWGS84_New
from OptimalInterpolation import get_data_path



# TODO: tidy up below
# TODO: add docstrings

class DataLoader():
    """ simple class to load gridded Free Board data"""

    def __init__(self,
                 sat_list=None,
                 seasons=None,
                 grid_res=None,
                 verbose=1):
        self.verbose = verbose

        #
        self.aux = None
        self.obs = None
        self.sie = None
        self.raw_obs = None

        if sat_list is None:
            sat_list = ["CS2_SAR", "CS2_SARIN", "S3A", "S3B"]
            if verbose > 1:
                print(f"sat_list not provided, using default: {sat_list}")
        self.sat_list = sat_list

        if seasons is None:
            seasons = ["2018-2019"]
            if verbose > 1:
                print(f"seasons not provided, using default: {seasons}")
        self.seasons = seasons

        if grid_res is None:
            grid_res = "25km"
            if verbose > 1:
                print(f"grid_res not provided, using default: {grid_res}")
        self.grid_res = grid_res

        # if verbose:
        #     print("")

    def _set_list_defaults(self, vals, defaults=None, var_name=""):
        # if vals is None then use some defaut
        if vals is None:
            vals = defaults
            if self.verbose:
                print(f"{var_name} not provide, will use default")
        # require vals be a list
        vals = vals if isinstance(vals, list) else [vals]
        if self.verbose:
            print(f"using: {vals}")
        return vals


    def load_obs_data(self,
                      sat_data_dir=None,
                      sat_list=None,
                      season="2018-2019",
                      grid_res=None,
                      take_dim_intersect=True):
        # TODO: only allow for one season at a time

        # get the grid_res to use
        grid_res = self.grid_res if grid_res is None else grid_res

        if sat_data_dir is None:
            # HARDCODED: path for data
            sat_data_dir = get_data_path("CS2S3_CPOM")
            if self.verbose:
                print(f"data_dir not provided, using default: {sat_data_dir}")

        # satellite list
        sat_list = sat_list if sat_list is not None else self.sat_list

        # TODO: here allow for reading of raw data, and setting as obs instead of the below
        sats = {}
        for s in sat_list:
            if self.verbose:
                print(f"reading in sat data for: {s}")
            pkl_file = f"{s}_dailyFB_{grid_res}_{season}_season.pkl"
            pkl_full = os.path.join(sat_data_dir, pkl_file)
            if not os.path.exists(pkl_full):
                print(f"FOR SAT: {s}, file not found: {pkl_file}, skipping")
                continue

            with open(pkl_full, "rb") as f:
                _ = pickle.load(f)
            # TODO: want to ensure dates are in sorted order
            _ = self._concat_dict_date_data(_)
            sats[s] = DataDict(vals=_['data'], dims=_["dims"], name=s)

        # take intersection of dates
        if take_dim_intersect:
            # warnings.warn("for satellite data taking intersection of dims")
            print("for satellite data taking intersection of dims")
            # {k: len(v.dims['date']) for k,v in self.sats.items()}
            int_dims = DataDict.dims_intersection(*[v.dims for k, v in sats.items()])
            for k in sats.keys():
                org_dims = {kk: len(vv) for kk, vv in sats[k].dims.items()}
                print(f"{k} - original dim sizes: {org_dims}")
                sats[k] = sats[k].subset(select_dims=int_dims)
            print("new_dims (sizes)")
            print({k: len(v) for k, v in int_dims.items()})

        self.sats = sats
        self.obs = DataDict.concatenate(*[v for v in sats.values()],
                                        dim_name="sat",
                                        name="obs")

    def load_raw_data(self,
                      raw_data_dir=None,
                      sat_list=None,
                      season="2018-2019"):

        # NOTE: raw data is assumed to be only for 2018-2019 season
        assert season=="2018-2019", f"currently only implemented for season '2018-2019'"
        # ----
        # read data
        # ---

        if raw_data_dir is None:
            # HARDCODED: path for data
            raw_data_dir = get_data_path("RAW")
            if self.verbose:
                print(f"raw_data_dir not provided, using default: {raw_data_dir}")

        # satellite list
        sat_list = sat_list if sat_list is not None else self.sat_list
        # sats = ["S3A", "S3B", "CS2_SAR", "CS2_SARIN"]

        sat_data = {}
        print("reading in (raw / along track) satellite data from:")
        for k in sat_list:
            print(k)
            sat_path = os.path.join(raw_data_dir, f"raw_along_track_lon_lat_fb_{k}.csv")
            assert os.path.exists(sat_path), f"sat file:\n{sat_path}\ndoes not exist!"

            _ = pd.read_csv(sat_path)
            # convert date datetime
            # TODO: here would like to convert to datetime - i.e. date with fraction of day
            #  - keep one days difference equal to 1
            # _['date'] = pd.to_datetime(_['date'].astype(str), format='%Y%m%d')
            # _['datetime'] = _['datetime'].astype('datetime64[s]')

            # convert lon, lat to x,y values - to be used for differences
            _['x'], _['y'] = WGS84toEASE2_New(_['lon'].values, _['lat'].values)

            sat_data[k] = DataDict.from_dataframe(_,
                                                  val_col='fb',
                                                  idx_col=['x', 'y', 'datetime'],
                                                  name=k)

            # HACK: keep dates as datetime64[D] - for now
            # TODO: allow this to be datetime
            sat_data[k].dims['datetime'] = sat_data[k].dims['datetime'].astype('datetime64[s]')

        tmp = [v for k, v in sat_data.items()]

        # store
        self.raw_obs = DataDict.concatenate(*tmp, dim_name="sat", name="obs")


    def load_aux_data(self,
                      aux_data_dir=None,
                      grid_res=None,
                      season=None,
                      load_sie=True,
                      load_fyi=True,
                      get_new=False,
                      get_bin_center=True,
                      legacy_projection=False):
        # TODO: set a default dir

        assert season is not None, f"season not provided, please provided"

        # get the grid_res to use
        grid_res = self.grid_res if grid_res is None else grid_res

        if self.verbose:
            print(f"loading 'aux' data for season='{season}', grid_res='{grid_res}'")

        # get the seasons to use
        # seasons = self.seasons if seasons is None else seasons
        # seasons = seasons if isinstance(seasons, list) else [seasons]

        # require data dir exists
        assert os.path.exists(aux_data_dir), f"aux_data_dir:\n{aux_data_dir}\ndoes not exist"

        # x,y, lon, lat points (used during the binning process)
        if self.verbose:
            print("reading 'aux' data")
        prefix_list = ["lon", "lat", "x", "y"]
        pre_prefix = "new_" if get_new else ""

        aux_dict = {}
        for p in prefix_list:
            try:
                aux_dict[p] = np.load(os.path.join(aux_data_dir, f"{pre_prefix}{p}_{grid_res}.npy"))
            except Exception as e:
                print(f"issue reading aux data with prefix: {p}\nError: {e}")
        self.aux = aux_dict
        # self.aux = {p: np.load(os.path.join(aux_data_dir, f"{pre_prefix}{p}_{grid_res}.npy"))
        #             for p in prefix_list}
        self.aux = {k: DataDict(vals=v, default_dim_name="grid_loc_")
                    for k, v in self.aux.items()}

        if get_bin_center:
            assert not get_new, f"get_bin_center is :{get_bin_center} but so is get_new: {get_new}, " \
                                f"which expects to read centered data"
            self.aux = self.move_to_bin_center_get_lon_lat(self.aux['x'].vals,
                                                           self.aux['y'].vals,
                                                           legacy_projection=legacy_projection)

        if load_sie:
            self.load_sie_data(sie_data_dir=os.path.join(aux_data_dir, "SIE"),
                               grid_res=grid_res,
                               season=season)

        if load_fyi:
            self._read_fyi(aux_data_dir, grid_res)


    def load_data(self,
                  aux_data_dir=None,
                  sat_data_dir=None,
                  raw_data_dir=None,
                  sat_list=None,
                  grid_res=None,
                  season=None,
                  **kwargs):

        if isinstance(grid_res, (int, float)):
            grid_res = f"{int(grid_res)}km"

        self.load_aux_data(aux_data_dir=aux_data_dir,
                           season=season,
                           grid_res=grid_res,
                           load_sie=True,
                           load_fyi=True,
                           **kwargs)

        self.load_obs_data(sat_data_dir=sat_data_dir,
                           sat_list=sat_list,
                           season=season,
                           grid_res=grid_res)

        if raw_data_dir is not None:
            self.load_raw_data(raw_data_dir=raw_data_dir)
        else:
            print("not loading raw data (raw_data_dir is None)")



    def _read_fyi(self, aux_data_dir, grid_res):

        if self.verbose:
            print("reading CS2_FYI data")
        if str(grid_res):
            grid_res = int(re.sub("km", "", grid_res))

        # TODO: deal with FYI being 181x181 (for 50)
        # - why is that the case

        # try/except needed because there is no 50km FYI data - use 25km instead
        try:
            cs2_FYI = np.load(
                os.path.join(aux_data_dir,
                             f'CS2_{grid_res}km_FYI_20181101-20190428.npy'))
        except FileNotFoundError:
            if (grid_res == 50) | (grid_res == '50'):
                print(f"grid_res: {grid_res} FYI not found, will try grid_res: 25")
            cs2_FYI = np.load(
                os.path.join(aux_data_dir,
                             f'CS2_25km_FYI_20181101-20190428.npy'))

        # create an array of dates
        cs2_FYI_dates = np.arange(np.datetime64("2018-11-01"), np.datetime64("2019-04-29"))
        cs2_FYI_dates = np.array([re.sub("-", "", i) for i in cs2_FYI_dates.astype(str)])

        self.fyi = DataDict(cs2_FYI, default_dim_name="grid_loc_")
        self.fyi.set_dim_idx(dim_idx="grid_loc_2", new_idx="date", dim_vals=cs2_FYI_dates)

    @classmethod
    def move_to_bin_center_get_lon_lat(self, x, y, legacy_projection=False):
        xnew = self.center_of_grid(x)
        ynew = self.center_of_grid(y)

        if legacy_projection:
            m = grid_proj(lon_0=360)
            lon, lat = m(xnew, ynew, inverse=True)
        else:
            lon, lat = EASE2toWGS84_New(xnew, ynew)

        out = {
            "lon": DataDict(lon, default_dim_name="grid_loc_"),
            "lat": DataDict(lat, default_dim_name="grid_loc_"),
            "x": DataDict(xnew, default_dim_name="grid_loc_"),
            "y": DataDict(ynew, default_dim_name="grid_loc_")
        }
        return out

    @staticmethod
    def center_of_grid(z):
        # given a 2-d array representing values on corners of grid
        # get the values in the center of grid
        # - should be equivalent to bilinear interpolation on an even spaces grid
        _ = z[:, :-1] + np.diff(z, axis=1) / 2
        out = _[:-1, :] + np.diff(_, axis=0) / 2
        return out

    def load_sie_data(self,
                      sie_data_dir=None,
                      grid_res=None,
                      season=None):

        assert season is not None, f"season not provided, please provided"

        # get the grid_res to use
        grid_res = self.grid_res if grid_res is None else grid_res

        # TODO: maybe don't have this nested, i.e. put in seperate function?
        # sie_data_dir = os.path.join(aux_data_dir, "SIE")
        assert os.path.exists(sie_data_dir), f"sie_data_dir:\n{sie_data_dir}\ndoes not exist"

        # ---
        # sea ice extent (sie)
        # ---
        if self.verbose:
            print("reading 'sie' data")
        with open(os.path.join(sie_data_dir, f"SIE_masking_{grid_res}_{season}_season.pkl"), "rb") as f:
            _ = pickle.load(f)
        _ = self._concat_dict_date_data(_)
        self.sie = DataDict(vals=_['data'], dims=_['dims'], name="sie")

    @staticmethod
    def coarse_grid(grid_space, grid_space_offset=0, x_size=320, y_size=320):
        # create a 2D array of False except along every grid_space points
        # - can be used to select a subset of points from a grid
        # NOTE: first dimension is treated as y dim
        cb_y, cb_x = np.zeros(x_size, dtype='bool'), np.zeros(y_size, dtype='bool')
        cb_y[(np.arange(len(cb_y)) % grid_space) == grid_space_offset] = True
        cb_x[(np.arange(len(cb_x)) % grid_space) == grid_space_offset] = True
        return cb_y[:, None] * cb_x[None, :]

    @staticmethod
    def data_select(date, dates, obs, xFB, yFB, days_ahead=4, days_behind=4):
        # for a given date select
        # print("selecting data")

        # find day location based on given date value
        day = np.where(np.in1d(dates, date))[0][0]

        # check the days about the window
        assert (day - days_behind) >= 0, \
            f"date:{date}, days_behind:{days_behind} gives day-days_behind: {day - days_behind}, which must be >= 0"

        assert (day + days_ahead) <= (len(dates) - 1), \
            f"date:{date}, days_ahead:{days_ahead} gives day+days_behind= {day + days_ahead}, " \
            f"which must be <= (len(dates) - 1) = {len(dates) - 1}"

        # the T days of training data from all satellites
        sat = obs[:, :, :, (day - days_behind):(day + days_ahead + 1)]

        # select data
        x_train, y_train, t_train, z = split_data2(sat, xFB, yFB)

        return x_train, y_train, t_train, z

    def _concat_dict_date_data(self, d):
        # dict keys assumed to dates, in YYYYMMDD format
        dates = np.array(list(d.keys()))
        # expect each date to be a nxn array
        # - add dim (for date) and concat
        out = np.concatenate([v[..., None] for k, v in d.items()], axis=-1)

        # sort dates - to be safe?
        # sort_dates = np.argsort(dates)

        # get x,y dimension values, if possible
        try:
            x = self.aux['x'].vals[0, :out.shape[1]]
            y = self.aux['y'].vals[:out.shape[0], 0]
        except Exception as e:
            if self.verbose:
                print("issue getting x,y values from aux data")
                print(e)
            x = np.arange(out.shape[1])
            y = np.arange(out.shape[0])

        # make a dimension dict
        dims = {
            "y": y,
            "x": x,
            "date": dates
        }

        return {"data": out, "dims": dims}

    def readFB(self, **kwargs):
        """wrapper for readFB function"""
        return readFB(**kwargs)

    @staticmethod
    def _parse_file_name(fn):
        _ = fn.split("_")
        out = {
            "date": _[1],
            "grid_res": _[2],
            "window": _[3],
            "radius": _[5]
        }
        return out

    def read_previous_data(self,
                           pkl_dir,
                           pkl_regex="\.pkl$",
                           parse_file_name_func=None,
                           get_bin_center=False):
        # TODO: allow for only selected data to be read in, i.e. just 'interp' or 'ell_x'

        # read previously generated (pickle) data in
        pfiles = [i for i in os.listdir(pkl_dir) if re.search(pkl_regex, i)]

        # function to parse attributes (date specifically) from (pickle) file name
        if parse_file_name_func is None:
            parse_file_name_func = self._parse_file_name

        # get the date from the file name
        date_file = {parse_file_name_func(pf)['date']: pf
                     for pf in pfiles}

        # get the sorted dates - assumes dates can be sorted, i.e. in YYYYMMDD format
        sort_dates = np.sort(np.array([k for k in date_file.keys()]))

        # store data in dict
        res = {}

        # select_data = select_data

        # read data in
        if self.verbose:
            print("reading in previously generated (legacy) interpolation results data")
        for sd in sort_dates:
            if self.verbose > 2:
                print(sd)
            pf = date_file[sd]
            with open(os.path.join(pkl_dir, pf), "rb") as f:
                fb = pickle.load(f)

            for k, v in fb.items():

                data_name = "_".join(k.split("_")[1:])
                if data_name in res:
                    res[data_name] = np.concatenate([res[data_name], v[..., None]], axis=-1)
                else:
                    # add a dimension, corresponding to date
                    res[data_name] = v[..., None]

        # add a dims dict
        # HARDCODED: x,y values - should probably read them from file
        res['dims'] = {
            "y": 2.5e4 + np.arange(0, 8e6, 5e4) if get_bin_center else np.arange(0, 8e6, 5e4),
            "x": 2.5e4 + np.arange(0, 8e6, 5e4) if get_bin_center else np.arange(0, 8e6, 5e4),
            "date": sort_dates
        }

        # convert to DataDict
        for k in res.keys():
            if k == 'dims':
                continue
            res[k] = DataDict(vals=res[k], dims=res['dims'], name=k)

        return res


    def interpolate_previous_data(self,
                                  res,
                                  xnew,
                                  ynew,
                                  keys_to_interp=None):
        # TODO: add doc string
        # TODO: add an option to only take points where there is sie
        # TODO: the mapping to a different projection should be reviewed
        #
        if keys_to_interp is None:
            if self.verbose:
                print("no keys_to_interp provided, getting all")
            keys_to_interp = [k for k in res.keys() if k != "dims"]

        for k in keys_to_interp:
            assert k in res.keys(), \
                f"in keys_to_interp there was {k} which is not available\navailable keys:\n{list(res.keys())}"

        # make a meshgrid for the current points
        x_, y_ = np.meshgrid(res['dims']['x'], res['dims']['y'])

        # map (old) points to lon, lat
        m = grid_proj(lon_0=360)
        ln, lt = m(x_, y_, inverse=True)

        # old points on new x,y grid
        xn, yn = WGS84toEASE2_New(ln, lt)

        # pairwise points to interpolate on
        xi = np.concatenate([xnew.flatten()[:, None], ynew.flatten()[:, None]], axis=1)

        # store output in a dict
        out = {}

        for k in keys_to_interp:
            if self.verbose:
                print(k)
            # select the data to be interpolated
            kdata = res[k]

            # create an array to populate with values
            out[k] = np.full(xnew.shape + (kdata.vals.shape[-1],), np.nan)

            # for id in range(kdata.shape[-1]):
            for id, date in enumerate(kdata.dims['date']):

                # data = kdata[..., id]
                data = kdata.subset(select_dims={"date": date})
                data = np.squeeze(data.vals)

                # identify the missing / nan points, just to exclude those
                has_val = ~np.isnan(data)

                # TODO: review this function
                z = griddata(points=(xn[has_val], yn[has_val]),
                             values=data[has_val],
                             xi=xi,
                             method='linear',
                             fill_value=np.nan)

                # reshape the output to be 2d
                z = z.reshape(xnew.shape)

                out[k][..., id] = z

        # store dimenson data
        out['dims'] = {
            "y":  ynew[:, 0],
            "x": xnew[0, :],
            "date": res['dims']['date']
        }

        # convert to DataDict
        for k in out.keys():
            if k == "dims":
                continue
            out[k] = DataDict(vals=out[k], dims=out['dims'], name=k)

        return out

    @staticmethod
    def _compare_dicts(d0, d1, chk_keys=None, verbose=False):

        if chk_keys is None:
            print("checking all keys")
            chk_keys = list(d0.keys())

        keys_matched = True
        for ck in chk_keys:

            try:
                if d0[ck] != d1[ck]:
                    print(f"key: {ck} did not match")
                    keys_matched = False
            except Exception as e:
                print(f"error on key: {ck}\nError:\n{e}")
                keys_matched = False
        if verbose & keys_matched:
            print("all keys matched")
        return keys_matched

    def read_results(self, results_dir,
                     file="results.csv",
                     data_cols=None,
                     attr_cols=None, grid_res_loc=None,
                     grid_size=360, unflatten=True, dates=None,
                     file_suffix=""):
        if self.verbose:
            print(f"reading previously generated outputs from:\n{results_dir}\nfrom files:\n{file}")
        # assert file in ["results.csv", "prediction.csv"], f"file: {file} not valid"
        # assert file in os.listdir(results_dir), f"file: {file} is not in results_dir:\n{results_dir}"

        if data_cols is not None:
            print("data_cols provided but it is not used")
            # if self.verbose:
            #     print("data_cols not provided, getting default values ")
            # if re.search("^results", file):
            #     data_cols = ['x', 'y', 'lon', 'lat',
            #                  'num_inputs', 'optimise_success',# "run_time",
            #                  "marginal_loglikelihood",
            #                  "ls_x", "ls_y", "ls_t",
            #                  "kernel_variance", "likelihood_variance",
            #                  ]
            # # otherwise expect predictions
            # else:
            #     # data_cols =['f*', 'f*_var', 'y', 'y_var', 'grid_loc_0', 'grid_loc_1', 'proj_loc_0', 'proj_loc_1',
            #     #             'mean', 'date', 'xs_x', 'xs_y', 'xs_t', 'run_time']
            #     data_cols = ['f*', 'f*_var', 'y', 'y_var', 'mean',
            #                  'xs_x', 'xs_y', 'xs_t', #'run_time'
            #                  ]

        assert os.path.exists(results_dir)

        # read in the input_config found in the results_dir (top level)
        config_file = os.path.join(results_dir, f"input_config{file_suffix}.json")
        assert os.path.exists(config_file), f"input_config{file_suffix}.json not found in results_dir:\n{results_dir}"

        with open(config_file, "r") as f:
            config = json.load(f)

        chk_keys = [k for k in config.keys() if k not in ["dates", "date", "run_info"]]

        # get the dates in results dir - i.e. the date stamped directories
        date_dirs = np.sort([i for i in os.listdir(results_dir) if i.isdigit()])

        assert len(date_dirs), f"no date_dirs found in result_dir"

        res_list = []

        if self.verbose:
            print("reading in previous results")

        for idx, date in enumerate(date_dirs):

            if dates is not None:
                if date not in dates:
                    print(f"skipping date: {date} because it's not in provided set: {dates}")
                    continue

            # check the input_config file
            date_conf_file = os.path.join(results_dir, date, f"input_config{file_suffix}.json")

            with open(date_conf_file, "r") as f:
                dconf = json.load(f)

            if not self._compare_dicts(config, dconf, chk_keys):
                print(f"some keys did not match for date: {date}, will skip")
                continue

            # check commit hash
            # - this could be to strict? however would expect results to be generated at same time
            comm0 = config['run_info']['git_info']['details'][0]
            comm1 = dconf['run_info']['git_info']['details'][0]
            if (comm0 != comm1):
                print(f"commits did not match, skipping date: {date}")
                continue

            print(date)
            # if file == "results.csv":
            if re.search("^results", file):
                print("reading a 'results' file")
                try:
                    # results contains attributes / parameters
                    res = pd.read_csv(os.path.join(results_dir, date, file))

                    # HACK: x,y (location) values had names change to x_loc, y_loc
                    #  - however some down stream functions expect to have x,y
                    res.rename(columns={"x_loc": "x", "y_loc": "y"}, inplace=True)

                    # HACK: check if some rows are all nan
                    if res.isnull().all(axis=1).any():
                        print("some rows had all nan")
                        res = res.loc[~res.isnull().all(axis=1)]
                        int_col = ['date', 'grid_loc_0', 'grid_loc_1']
                        for ic in int_col:
                            res[ic] = res[ic].astype(int)

                    res_list.append(res)
                except FileNotFoundError:
                    print(f"file: {file} not found, skipping")
                    continue
            else:
                print("reading a 'predictions' file")
                try:
                    pred = pd.read_csv(os.path.join(results_dir, date, file))
                    # HACK: this shouldn't need to be done, change data at source instead of here
                    pred.rename(columns={c: c.lstrip() for c in pred.columns}, inplace=True)

                    pred = pred.loc[(pred['grid_loc_0'] == pred['proj_loc_0']) & \
                                    (pred['grid_loc_1'] == pred['proj_loc_1'])]

                    res_list.append(pred)
                except FileNotFoundError:
                    print("prediction.csv file not found, skipping")
                    continue

        rdf = pd.concat(res_list)
        rdf['date'] = rdf['date'].astype(str)

        # if data_cols is not None:
        #     miss_data_cols = np.array(data_cols)[~np.in1d(data_cols, rdf.columns)]
        #
        #     assert len(miss_data_cols) == 0, f"data_cols: {miss_data_cols} not in data"

        dim_cols = ["grid_loc_0", "grid_loc_1", "date"]
        dims = {dc: rdf[dc].values for dc in dim_cols}

        dd = {dc: DataDict(vals=rdf[dc].values, dims=dims, is_flat=True, name=dc)
              for dc in rdf.columns if dc not in dims }

        # unflatten data - put into cube
        if unflatten:

            udims = {k: np.sort(np.unique(v)) for k, v in dims.items()}

            # HACK: make grid locations align with data
            if grid_res_loc:
                if self.verbose:
                    print(f"grid_res_loc: {grid_res_loc} provided will hard coded grid_loc_# values")
                udims['grid_loc_0'] = np.arange(grid_size) if grid_res_loc == 25 else np.arange(grid_size/2)
                udims['grid_loc_1'] = np.arange(grid_size) if grid_res_loc == 25 else np.arange(grid_size/2)

            for k in dd.keys():
                dd[k] = dd[k].unflatten(udims=udims)

        if attr_cols is None:
            if self.verbose:
                print("using default attr_cols")
            if re.search("^results", file):
                attr_cols = ["scale_x", "scale_y", "scale_t", "scale_output"]
            else:
                attr_cols = []

        attr_vals = {}
        for ac in attr_cols:
            _ = rdf[ac].unique()
            assert len(_) == 1, f"got multiple values for attribute column: {ac}"
            attr_vals[ac] = _[0]

        # TODO: add attributes to just the relevant DataDicts (?)
        # add attributes
        for k, v in attr_vals.items():
            for kk in dd.keys():
                dd[kk][k] = v

        # add input config as attribute
        # for kk in dd.keys():
        #     dd[kk]['config'] = config

        # store as input_config
        dd['input_config'] = config

        return dd


if __name__ == "__main__":

    dl = DataLoader()

    # load EASE data - expects folder structure: <package_dir>/data/EASE/freeboard_daily_processed/CS2S3_CPOM

    # specify data directories
    sat_data_dir = get_data_path("CS2S3_CPOM")
    aux_data_dir = get_data_path("aux")
    raw_data_dir = get_data_path("RAW")

    # load aux data
    dl.load_aux_data(aux_data_dir=aux_data_dir, grid_res="50km", season="2018-2019")
    # load sat data
    dl.load_obs_data(sat_data_dir=sat_data_dir, grid_res="50km", season="2018-2019")

    # data is store in a dict with keys 'data' and 'dims'
    aux, sie, obs = dl.aux, dl.sie, dl.obs

    # load raw data
    dl.load_raw_data(raw_data_dir=raw_data_dir)

