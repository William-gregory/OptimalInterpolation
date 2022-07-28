# example of running sea ice freeboard interpolation (for a given date)

import time
import os
import json
import datetime
import subprocess
import numpy as np
import pandas as pd

from OptimalInterpolation import get_data_path, get_path
from OptimalInterpolation.sea_ice_freeboard import SeaIceFreeboard
from OptimalInterpolation.utils import EASE2toWGS84_New, get_git_information, move_to_archive

if __name__ == "__main__":

    output_base_dir = get_path("results")

    # ---
    # input config
    # ---

    config = {
        "dates": ["20181201"],  # , "20190101", "20190201", "20190301"],
        "optimise": False,
        "file_suffix": "_chkparams",
        "output_dir": os.path.join(output_base_dir, "test_svgp"),
        "inclusion_radius": 300,
        "days_ahead": 4,
        "days_behind": 4,
        "data_dir": "package",
        "season": "2018-2019",
        "grid_res": 50,
        "coarse_grid_spacing": 5,
        "min_inputs": 5,
        "verbose": 1,
        "engine": "GPflow_svgp",
        "kernel": "Matern32",
        # "mean_function": "constant",
        "hold_out": ["S3B"],
        "load_params": True,
        "overwrite": True,
        "predict_on_hold": True,
        "scale_inputs": True,
        "scale_outputs": False,
        "bound_length_scales": False,
        "append_to_file": True,
        "post_process": {
            "prev_results_dir": os.path.join(output_base_dir, "test_svgp","radius300_daysahead4_daysbehind4_gridres50_season2018-2019_coarsegrid5_holdoutS3B_boundlsFalse"),
            "prev_results_file": None,
            "prev_file_suffix": "",
            "clip_and_smooth": False,
            "vmax_map": {
                "ls_x": 2 * 300 * 1000,
                "ls_y": 2 * 300 * 1000,
                "ls_t": 9,
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
        },
        "inducing_point_params": {
            "num_inducing_points": 50,
            "min_obs_for_svgp": 500
        }
    }

    # ---
    # parameters
    # ---

    print("using config:")
    print(json.dumps(config, indent=4))

    # extract parameters from config

    season = config.get("season", "2018-2019")
    assert season == "2018-2019", "only can handle data inputs from '2018-2019' season at the moment"

    dates = config['dates']
    output_dir = config['output_dir']
    optimise = config.get('optimise', False)
    days_ahead = config.get("days_ahead", 4)
    days_behind = config.get("days_behind", 4)
    season = config.get("season", "2018-2019")
    data_dir = config.get("data_dir", "package")

    incl_rad = config.get("inclusion_radius", 300)
    grid_res = config.get("grid_res", 25)
    coarse_grid_spacing = config.get("coarse_grid_spacing", 1)
    min_inputs = config.get("min_inputs", 10)
    # min sea ice cover - when loading data set sie to nan if < min_sie
    min_sie = config.get("min_sie", 0.15)

    engine = config.get("engine", "GPflow")
    kernel = config.get("kernel", "Matern32")
    overwrite = config.get('overwrite', False)
    hold_out = config.get("hold_out", None)

    scale_inputs = config.get("scale_inputs", False)
    scale_inputs = [1 / (grid_res * 1000), 1 / (grid_res * 1000), 1.0] if scale_inputs else [1.0, 1.0, 1.0]

    scale_outputs = config.get("scale_outputs", False)
    scale_outputs = 100. if scale_outputs else 1.

    append_to_file = config.get("append_to_file", True)

    # use holdout location as GP location selection criteria (in addition to coarse_grid_spacing, etc)
    pred_on_hold_out = config.get("predict_on_hold", True)

    bound_length_scales = config.get("bound_length_scales", True)

    mean_function = config.get("mean_function", None)
    file_suffix = config.get("file_suffix", "")
    post_process_config = config.get("post_process", {})

    load_params = config.get("load_params", False)

    inducing_point_params = config.get("inducing_point_params", {})

    # -----
    # initialise SeaIceFreeboard object
    # -----

    sifb = SeaIceFreeboard(grid_res=f"{grid_res}km",
                           length_scale_name=["x", "y", "t"])

    # ---
    # read / load data
    # ---

    if data_dir == "package":
        data_dir = get_data_path()

    assert os.path.exists(data_dir)

    sifb.load_data(aux_data_dir=os.path.join(data_dir, "aux"),
                   sat_data_dir=os.path.join(data_dir, "CS2S3_CPOM"),
                   # raw_data_dir=os.path.join(data_dir, "RAW"),
                   season=season)

    for date in dates:
        print("#" * 100)
        print(f"date: {date}")
        print("#" * 10)

        sifb.run(date=date,
                 output_dir=output_dir,
                 days_ahead=days_ahead,
                 days_behind=days_behind,
                 incl_rad=incl_rad,
                 grid_res=grid_res,
                 coarse_grid_spacing=coarse_grid_spacing,
                 min_inputs=min_inputs,
                 min_sie=min_sie,
                 engine=engine,
                 kernel=kernel,
                 overwrite=overwrite,
                 load_params=load_params,
                 optimise=optimise,
                 season=season,
                 hold_out=hold_out,
                 scale_inputs=scale_inputs,
                 scale_outputs=scale_outputs,
                 append_to_file=append_to_file,
                 pred_on_hold_out=pred_on_hold_out,
                 bound_length_scales=bound_length_scales,
                 mean_function=mean_function,
                 file_suffix=file_suffix,
                 post_process=post_process_config,
                 print_every=5,
                 inducing_point_params=inducing_point_params)

