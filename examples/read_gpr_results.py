# read in previously generated GPR hyper parameters and results
# - results generated from SeaIceFreeboard methods

import pandas as pd
import os
from OptimalInterpolation.sea_ice_freeboard import SeaIceFreeboard
from OptimalInterpolation import get_path, read_key_from_config

pd.set_option("display.max_columns", 200)

# ---
# read in previously generated data (and interpolate to EASE grid)
# ---

sifb = SeaIceFreeboard()

# ----
# read gpflow generated data
# ----

grid_res = 25

# --
# results_dir: change as needed
# --

# google drive location
gdrive = read_key_from_config("directory_locations", "gdrive",
                              example="gdrive")
base_dir = os.path.join(gdrive, "Dissertation/refactored")
sub_dir = f"radius300_daysahead4_daysbehind4_gridres{grid_res}_season2018-2019_coarsegrid1_holdout_boundlsFalse"
results_dir = os.path.join(base_dir, sub_dir)


# --
# read results - hyper parameters values, log likelihood
# --

res = sifb.read_results(results_dir,
                        file="results.csv",
                        grid_res_loc=grid_res,
                        unflatten=True)

# --
# read predictions - f*, f*_var
# --


pre = sifb.read_results(results_dir,
                        file="prediction.csv",
                        grid_res_loc=grid_res,
                        unflatten=True)



