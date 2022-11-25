
# make predictions for radar freeboard for a given date and set of configuration parameters
# - using raw observations
# - results are written to file specified by output_dir/tmp_dir

import os
from OptimalInterpolation import get_data_path, get_path
from OptimalInterpolation.sea_ice_freeboard import SeaIceFreeboard

output_dir = get_path("results")
if not os.path.exists(output_dir):
    print(output_dir)
    print("does not exists, creating")
    os.makedirs(output_dir, exist_ok=True)


# ---
# Parameters
# ---

# parameters that define 'tmp_dir'
grid_res = 50
incl_rad = 300
days_ahead = 4
days_behind = 4
bound_length_scales = True
prior_mean_method = "fyi_average"

engine = "GPflow_svgp"

verbose = 3
season = "2018-2019"
date = "20181201"

min_inputs = 500
num_inducing_points = 500
min_obs_for_svgp = 500
use_raw_data = True
initial_file_suffix = ""

# locations to make predictions
prediction_locations = [
        "center_only",
        # "obs_in_cell", # needs hold_out to be not None
        {"name": "evenly_spaced_in_cell", "n": 100}
    ]

# optimisation parameters for SVGP
optimise_params = {
    "use_minibatch": False,
    "gamma": 1.0,
    "learning_rate": 0.07,
    "trainable_inducing_variable": False,
    "minibatch_size": 2000,
    "maxiter": 150,
    "log_freq": 10,
    "persistence": 10,
    "early_stop": True,
    "save_best": False
}

# parameters for inducing points
inducing_point_params = {
    "num_inducing_points": num_inducing_points,
    "min_obs_for_svgp": min_obs_for_svgp,
    "inducing_locations": "random"
}


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
               raw_data_dir=os.path.join(data_dir, "RAW"),
               season=season)

# ---
# initial run
# ---

print("|*" * 50)
print(f"initial run")

# tmp_dir is where the results will be written to
tmp_dir = "using_raw_obs"

print(f"writing results to sub-dir:\n{tmp_dir}")
print("-" * 10)

# optimise - no post processing

sifb.run(date=date,
         output_dir=output_dir,
         optimise=True,
         engine=engine,
         file_suffix=initial_file_suffix,
         grid_res=grid_res,
         incl_rad=incl_rad,
         days_ahead=days_ahead,
         days_behind=days_behind,
         min_inputs=min_inputs,
         bound_length_scales=bound_length_scales,
         use_raw_data=True,
         optimise_params=optimise_params,
         inducing_point_params=inducing_point_params,
         predict_locations=prediction_locations,
         predict_in_neighbouring_cells=1,
         tmp_dir=tmp_dir)
