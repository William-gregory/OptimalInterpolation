# make predictions for radar freeboard for a given date and set of configuration parameters
# - hyper parameters are initially optimised and then smoothed in a post processing step
# - results are written to file specified by output_dir/tmp_dir


import os
from OptimalInterpolation import get_data_path, get_path
from OptimalInterpolation.sea_ice_freeboard import SeaIceFreeboard

# can check if will using a gpu by uncommenting: (also: gpu name will be given in results file)
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# print("GPUS")
# print(gpus)

#  change output_dir as needed
output_dir = get_path("results")
if not os.path.exists(output_dir):
    print(output_dir)
    print("does not exists, creating")
    os.makedirs(output_dir, exist_ok=True)

# ---
# Parameters
# ---

grid_res = 25
coarse_grid_spacing = 2
incl_rad = 300
days_ahead = 4
days_behind = 4
min_inputs = 5
bound_length_scales = True

prior_mean_method = "fyi_average"
season = "2018-2019"
assert season == "2018-2019", "only can handle data inputs from '2018-2019' season at the moment"
verbose = 3
date = "20181201"

# -----
# initialise SeaIceFreeboard object
# -----

sifb = SeaIceFreeboard(grid_res=f"{grid_res}km",
                       length_scale_name=["x", "y", "t"],
                       verbose=verbose,
                       rng_seed=None)

# ---
# read / load data
# ---

data_dir = get_data_path()
assert os.path.exists(data_dir)

sifb.load_data(aux_data_dir=os.path.join(data_dir, "aux"),
               sat_data_dir=os.path.join(data_dir, "CS2S3_CPOM"),
               season=season)

# ---
# for each date increment over the model combinations
# ---

print("|" * 100)
print(f"date: {date}")
print("#" * 10)

# ---
# initial run - optimise points before smoothing
# ---

# tmp_dir is where the results will be written to
# - can use sifb.make_temp_dir(...) to make string containing key parameters of run
tmp_dir = "some_basic_results"

# optimise - no post processing
sifb.run(date=date,
         output_dir=output_dir,
         optimise=True,
         grid_res=grid_res,
         incl_rad=incl_rad,
         days_ahead=days_ahead,
         days_behind=days_behind,
         min_inputs=min_inputs,
         coarse_grid_spacing=coarse_grid_spacing,
         bound_length_scales=bound_length_scales,
         predict_locations="center_only",
         file_suffix="",
         scale_inputs=True,
         tmp_dir=tmp_dir)

# ---
# apply post processing
# ---

print("|*" * 50)
print(f"post processing")
print(f"date: {date}")
print("#" * 10)

# for post processing - need results from previous run
previous_results = {
    'dir': os.path.join(output_dir, tmp_dir),
    'suffix': ""
}

# post processing dictionary
# std is used by the kernel smooth with units being grid spacing, i.e. adjacent grid cells have space of 1
post_process = {
    "clip_and_smooth": True,
    "smooth_method": "kernel",
    "std": coarse_grid_spacing,
    "vmax_map": {
        "ls_x": 2 * incl_rad * 1000,
        "ls_y": 2 * incl_rad * 1000,
        "ls_t": days_ahead + days_behind + 1,
        "kernel_variance": 0.1,
        "likelihood_variance": 0.05
    },
    "vmin_map": {
        "ls_x": 1,
        "ls_y": 1,
        "ls_t": 1e-6,
        "kernel_variance": 2e-6,
        "likelihood_variance": 2e-6
    }
}

# will write results using a different file_suffix
sifb.run(date=date,
         optimise=False,
         coarse_grid_spacing=1,
         file_suffix="_postproc",
         post_process=post_process,
         previous_results=previous_results,
         tmp_dir=tmp_dir)

