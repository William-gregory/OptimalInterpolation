# DataLoader Class

import numpy as np
import os
import re
import pickle

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
        self.aux = None,
        self.obs = None,
        self.sie = None

        # if sat_data_dir is None:
        #     sat_data_dir = get_data_path("CS2S3_CPOM")
        #     if verbose > 1:
        #         print(f"sat_data_dir not provided, using default: {sat_data_dir}")
        #
        # if data_dir is None:
        #     if verbose > 1:
        #         print("no data dir provided, will default to package")
        #     self.data_dir = get_data_path()
        # else:
        #     if verbose > 1:
        #         print(f"setting data_dir: {data_dir}")
        #     self.data_dir = data_dir

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
                      make_obs=True,
                      combine_all=True,
                      common_dates=True):
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

        self.sats = sats

        # {k: len(v.dims['date']) for k,v in self.sats.items()}
        # int_dims = DataDict.dims_intersection(*[v.dims for k,v in self.sats.items()])

        self.obs = DataDict.concatenate(*[v for v in sats.values()],
                                        dim_name="sat",
                                        name="obs")


        # # ---
        # # date alignment
        # # ---
        #
        # # TODO: allow for all dates to be returned, wont be able concatenate
        # # TODO: review this, getting the common dates does not seem needed
        # # take only the intersection
        #
        # if common_dates:
        #     sat_dates = {k: v.dims['date'] for k, v in sats.items()}
        #
        #     common_dates = reduce(lambda x, y: np.intersect1d(x, y),
        #                           [v for v in sat_dates.values()])
        #
        #     if self.verbose:
        #         print(f"found: {len(common_dates)} common_dates in the data, will use only these dates")
        #         if self.verbose > 1:
        #             for k, v in sats.items():
        #                 print(f"{k}: {len(v.dims['date'])}")
        #
        #     # for each dataset only take the dates which are common
        #     for k in sats.keys():
        #
        #         sats[k].subset(select_dims={"date": common_dates}, inplace=True)
        #         # v = sats[k]
        #         # date_bool = np.in1d(v.dims['date'], common_dates)
        #         # v['data'] = v['data'][..., date_bool]
        #         # v['dims']['date'] = v['dims']['date'][date_bool]
        #
        #
        # if combine_all:
        #     # sat_dim = np.array(list(sats))
        #     # concatenate data
        #     obs = DataDict.concatenate(*[v for v in sats.values()],
        #                                dim_name="sat",
        #                                name="obs")
        #     # tmp = np.concatenate([v[..., None]
        #     #                       for k, v in sats.items()], axis=-1)
        #     #
        #     #
        #     #
        #     # # just to be consistent with previous layout, will swap axis
        #     # tmp = tmp.swapaxes(2, 3)
        #     #
        #     # dims = {
        #     #     'y': sats[sat_dim[0]]['dims']['y'],
        #     #     'x': sats[sat_dim[0]]['dims']['x'],
        #     #     'sat': sat_dim,
        #     #     'date': sats[sat_dim[0]]['dims']['date']
        #     # }
        #
        #     self.obs = obs
        # else:
        #     self.obs = sats

    def load_aux_data(self,
                      aux_data_dir=None,
                      grid_res=None,
                      season=None,
                      load_sie=True,
                      load_fyi=True,
                      get_new=False,
                      get_bin_center=True):
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
        self.aux = {p: np.load(os.path.join(aux_data_dir, f"{pre_prefix}{p}_{grid_res}.npy"))
                    for p in prefix_list}
        self.aux = {k: DataDict(vals=v)
                    for k, v in self.aux.items()}

        if get_bin_center:
            assert not get_new, f"get_bin_center is :{get_bin_center} but so is get_new: {get_new}, " \
                                f"which expects to read centered data"
            self.aux = self.move_to_bin_center_get_lon_lat(self.aux['x'].vals,
                                                           self.aux['y'].vals)

        if load_sie:
            self.load_sie_data(sie_data_dir=os.path.join(aux_data_dir, "SIE"),
                               grid_res=grid_res,
                               season=season)

        if load_fyi:
            self._read_fyi(aux_data_dir, grid_res)


    def load_data(self,
                  aux_data_dir=None,
                  sat_data_dir=None,
                  sat_list=None,
                  grid_res=None,
                  season=None,
                  **kwargs):



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




    def _read_fyi(self, aux_data_dir, grid_res):

        if self.verbose:
            print("reading CS2_FYI data")
        if str(grid_res):
            grid_res = int(re.sub("km", "", grid_res))

        # TODO: deal with FYI being 181x181 (for 50)
        # - why is that the case
        cs2_FYI = np.load(
            os.path.join(aux_data_dir,
                         f'CS2_{grid_res}km_FYI_20181101-20190428.npy'))

        # create an array of dates
        cs2_FYI_dates = np.arange(np.datetime64("2018-11-01"), np.datetime64("2019-04-29"))
        cs2_FYI_dates = np.array([re.sub("-", "", i) for i in cs2_FYI_dates.astype(str)])

        self.fyi = DataDict(cs2_FYI)
        self.fyi.set_dim_idx(dim_idx="idx2", new_idx="date", dim_vals=cs2_FYI_dates)

    @classmethod
    def move_to_bin_center_get_lon_lat(self, x, y):
        xnew = self.center_of_grid(x)
        ynew = self.center_of_grid(y)

        lon, lat = EASE2toWGS84_New(xnew, ynew)
        out = {
            "lon": DataDict(lon),
            "lat": DataDict(lat),
            "x": DataDict(xnew),
            "y": DataDict(ynew)
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
                           parse_file_name_func=None):
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
            print("reading in data")
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
            "y":  np.arange(0, 8e6, 5e4),
            "x": np.arange(0, 8e6, 5e4),
            "date": sort_dates
        }

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
            out[k] = np.full(xnew.shape + (kdata.shape[-1],), np.nan)

            for id in range(kdata.shape[-1]):

                data = kdata[..., id]

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

        return out


if __name__ == "__main__":

    dl = DataLoader()

    # load EASE data - expects folder structure: <package_dir>/data/EASE/freeboard_daily_processed/CS2S3_CPOM

    # specify data directories
    sat_data_dir = get_data_path("CS2S3_CPOM")
    aux_data_dir = get_data_path("aux")

    # load aux data
    dl.load_aux_data(aux_data_dir=aux_data_dir, grid_res="50km", season="2018-2019")
    # load sat data
    dl.load_obs_data(sat_data_dir=sat_data_dir, grid_res="50km", season="2018-2019")

    # data is store in a dict with keys 'data' and 'dims'
    aux, sie, obs = dl.aux, dl.sie, dl.obs

