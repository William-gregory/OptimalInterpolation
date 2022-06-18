# DataLoader Class

import numpy as np
import os
import re
import pickle

from OptimalInterpolation.utils import readFB
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
                      grid_res="25km",
                      combine_all=True):
        # TODO: only allow for one season at a time

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

            sats[s] = self._concat_dict_date_data(_)

        # TODO: should have a double check the dates all align

        if combine_all:
            sat_dim = np.array(list(sats))
            # concatenate data
            tmp = np.concatenate([v['data'][..., None]
                                  for k, v in sats.items()], axis=-1)

            # just to be consistent with previous layout, will swap axis
            tmp = tmp.swapaxes(2, 3)

            dims = {
                'y': sats[sat_dim[0]]['dims']['y'],
                'x': sats[sat_dim[0]]['dims']['x'],
                'sat': sat_dim,
                'date': sats[sat_dim[0]]['dims']['date']
            }

            self.obs = {"data": tmp, "dims": dims}

        else:
            self.obs = sats

    def load_aux_data(self,
                      aux_data_dir=None,
                      grid_res=None,
                      season=None):
        # TODO: set a default dir

        assert season is not None, f"season not provided, please provided"

        # get the grid_res to use
        grid_res = self.grid_res if grid_res is None else grid_res

        # get the seasons to use
        # seasons = self.seasons if seasons is None else seasons
        # seasons = seasons if isinstance(seasons, list) else [seasons]

        # require data dir exists
        assert os.path.exists(aux_data_dir), f"aux_data_dir:\n{aux_data_dir}\ndoes not exist"

        # x,y, lon, lat points (used during the binning process)
        if self.verbose:
            print("reading 'aux' data")
        prefix_list = ["lon", "lat", "x", "y"]
        self.aux = {p: np.load(os.path.join(aux_data_dir, f"{p}_{grid_res}.npy"))
                    for p in prefix_list}

        # TODO: maybe don't have this nested, i.e. put in seperate function?
        sie_data_dir = os.path.join(aux_data_dir, "SIE")
        assert os.path.exists(sie_data_dir), f"sie_data_dir:\n{sie_data_dir}\ndoes not exist"

        # ---
        # sea ice extent (sie)
        # ---
        if self.verbose:
            print("reading 'sie' data")
        with open(os.path.join(sie_data_dir, f"SIE_masking_{grid_res}_{season}_season.pkl"), "rb") as f:
            _ = pickle.load(f)
        self.sie = self._concat_dict_date_data(_)

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
            x = self.aux['x'][0, :-1]
            y = self.aux['y'][:-1, 0]
        except Exception as e:
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


if __name__ == "__main__":


    dl = DataLoader()

    # load EASE data - expects folder structure: <package_dir>/data/EASE/freeboard_daily_processed/CS2S3_CPOM

    # specify data directories
    sat_data_dir = get_data_path("CS2S3_CPOM")
    aux_data_dir = get_data_path("aux")

    # load aux data
    dl.load_aux_data(aux_data_dir=aux_data_dir, season="2018-2019")
    # load sat data
    dl.load_obs_data(sat_data_dir=sat_data_dir, season="2018-2019")

    # data is store in a dict with keys 'data' and 'dims'
    aux, sie, obs = dl.aux, dl.sie, dl.obs