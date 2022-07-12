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

    t_total0 = time.time()

    output_base_dir = get_path()

    # ---
    # input config
    # ---

    # specify parameters in a config
    # - input config will be written to file
    config = {
        "dates": ["20181201", "20190101"],  # "20181201"
        "inclusion_radius": 300,
        "days_ahead": 4,
        "days_behind": 4,
        "data_dir": "package",
        "season": "2018-2019",
        "grid_res": 50,
        "coarse_grid_spacing": 4,
        "min_inputs": 10,
        "verbose": 1,
        "engine": "GPflow",
        "kernel": "Matern32",
        # "initialise_with_neighbors": True,
        "hold_out": ["S3B"],
        "predict_on_hold": True,
        "scale_inputs": True,
        "scale_outputs": True,
        "bound_length_scales": True,
        "append_to_file": True,
        "output_dir": os.path.join(output_base_dir, "local_results")
    }

    # ---
    # parameters
    # ---

    # extract parameters from config

    season = config.get("season", "2018-2019")
    assert season == "2018-2019", "only can handle data inputs from '2018-2019' season at the moment"

    dates = config['dates']
    days_ahead = config.get("days_ahead", 4)
    days_behind = config.get("days_behind", 4)
    incl_rad = config.get("inclusion_radius", 300)
    grid_res = config.get("grid_res", 25)
    coarse_grid_spacing = config.get("coarse_grid_spacing", 1)
    min_inputs = config.get("min_inputs", 10)
    # min sea ice cover - when loading data set sie to nan if < min_sie
    min_sie = config.get("min_sie", 0.15)

    engine = config.get("engine", "GPflow")
    kernel = config.get("kernel", "Matern32")
    hold_out = config.get("hold_out", None)

    scale_inputs = config.get("scale_inputs", False)
    scale_inputs = [1 / (grid_res * 1000), 1 / (grid_res * 1000), 1.0] if scale_inputs else [1.0, 1.0, 1.0]

    scale_outputs = config.get("scale_outputs", False)
    scale_outputs = 100. if scale_outputs else 1.

    append_to_file = config.get("append_to_file", True)

    # use holdout location as GP location selection criteria (in addition to coarse_grid_spacing, etc)
    pred_on_hold_out = config.get("predict_on_hold", True)

    bound_length_scales = config.get("bound_length_scales", True)
    # length scale bounds
    if bound_length_scales:
        # NOTE: input scaling will happen in model build (for engine: GPflow)
        ls_lb = np.zeros(len(scale_inputs))
        ls_ub = np.array([(2 * incl_rad * 1000),
                         (2 * incl_rad * 1000),
                         (days_behind + days_ahead + 1)])

    else:
        ls_lb, ls_ub = None, None

    # -
    # results dir
    # -

    results_dir = config["output_dir"]

    # subdirectory in results dir, with name containing (some) run parameters
    tmp_dir = f"radius{incl_rad}_daysahead{days_ahead}_daysbehind{days_behind}_gridres{grid_res}_season{season}_coarsegrid{coarse_grid_spacing}_holdout{'|'.join(hold_out)}_boundls{bound_length_scales}"
    output_dir = os.path.join(results_dir, tmp_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ---
    # run info
    # ---

    # run time info
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_info = {
        "run_datetime": now
    }

    # add git_info
    try:
        run_info['git_info'] = get_git_information()
    except subprocess.CalledProcessError:
        print("issue getting git_info, check current working dir")
        pass

    config["run_info"] = run_info

    # -----
    # initialise SeaIceFreeboard object
    # -----

    sifb = SeaIceFreeboard(grid_res=f"{grid_res}km",
                           length_scale_name=["x", "y", "t"])

    # ---
    # read / load data
    # ---

    sifb.load_data(aux_data_dir=get_data_path("aux"),
                   sat_data_dir=get_data_path("CS2S3_CPOM"),
                   season=season)

    # ----
    # for each date: select data used to build GP
    # ----

    all_res = []
    all_preds = []

    for date in dates:
        print(f"date: {date}")
        # --
        # date directory and file name
        # --

        date_dir = os.path.join(output_dir, date)
        os.makedirs(date_dir, exist_ok=True)

        # results will be written to file
        res_file = os.path.join(date_dir, "results.csv")
        pred_file = os.path.join(date_dir, "prediction.csv")
        # bad results will be written to
        skip_file = os.path.join(date_dir, "skipped.csv")

        # ---
        # move files to archive, if they already exist
        # ---

        move_to_archive(top_dir=date_dir,
                        file_names=["input_config.json",
                                    "results.csv",
                                    "prediction.csv",
                                    "skipped.csv"],
                        suffix=f"_{now}",
                        verbose=True)

        # ---
        # write config to file - will end up doing this for each date
        # ---

        with open(os.path.join(date_dir, "input_config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # --
        # select data for given date
        # --

        # select data for a given date (include some days ahead / behind)
        sifb.select_obs_date(date,
                             days_ahead=days_ahead,
                             days_behind=days_behind)

        # set values on date for hold_out (satellites) to nan
        sifb.remove_hold_out_obs_date(hold_out=hold_out)

        # calculate the mean for values obs
        sifb.prior_mean(date,
                        method="fyi_average")

        # de-mean the observation (used for the calculation on the given date)
        sifb.demean_obs_date()

        # build KD-tree
        sifb.build_kd_tree()

        # ----
        # locations to calculate GP on
        # ----

        gp_locs = sifb.select_gp_locations(date=date,
                                           min_sie=min_sie,
                                           coarse_grid_spacing=coarse_grid_spacing,
                                           sat_names=hold_out if pred_on_hold_out else None)
        select_loc = np.where(gp_locs)

        # ---
        # for each location to optimise GP
        # ---

        print(f"will calculate GPs at: {gp_locs.sum()} locations")
        sifb.verbose = 0
        for i in range(gp_locs.sum()):

            t0 = time.time()
            # ---
            # select location
            # ---

            grid_loc = select_loc[0][i], select_loc[1][i]
            x_ = sifb.aux['x'].vals[grid_loc]
            y_ = sifb.aux['y'].vals[grid_loc]

            # alternatively could use
            # x_ = sifb.obs.dims['x'][grid_loc[1]]
            # y_ = sifb.obs.dims['y'][grid_loc[0]]

            # select inputs for a given location
            inputs, outputs = sifb.select_input_output_from_obs_date(x=x_,
                                                                     y=y_)

            # too few inputs?
            if len(inputs) < min_inputs:
                print("too few inputs")
                tmp = pd.DataFrame({"grid_loc_0": grid_loc[0],
                                    "grid_loc_1": grid_loc[1],
                                    "reason": f"had only {len(inputs)} inputs"},
                                   index=[i])

                tmp.to_csv(skip_file, mode='a',
                           header=not os.path.exists(skip_file),
                           index=False)
                continue

            # ---
            # build a GPR model for data
            # ---

            sifb.build_gpr(inputs=inputs,
                           outputs=outputs,
                           scale_inputs=scale_inputs,
                           scale_outputs=scale_outputs,
                           length_scale_lb=ls_lb,
                           length_scale_ub=ls_ub,
                           engine=engine,
                           kernel=kernel)

            # ---
            # get the hyper parameters
            # ---

            hps = sifb.get_hyperparameters(scale_hyperparams=False)

            # ---
            # optimise model
            # ---

            opt_hyp = sifb.optimise(scale_hyperparams=False)

            t1 = time.time()

            # ---
            # make predictions
            # ---

            # TODO: predict in region around center point
            x_pred, y_pred, gl0, gl1 = sifb.get_neighbours_of_grid_loc(grid_loc,
                                                                       coarse_grid_spacing=coarse_grid_spacing)

            preds = sifb.predict_freeboard(x=x_pred,
                                           y=y_pred)

            preds['grid_loc_0'] = grid_loc[0]
            preds['grid_loc_1'] = grid_loc[1]
            preds["proj_loc_0"] = gl0
            preds["proj_loc_1"] = gl1
            # TODO: this needs to be more robust to handle different mean priors
            preds['mean'] = sifb.mean.vals[gl0, gl1]
            preds['date'] = date

            # the split the test values per dimension
            xs = preds.pop('xs')
            for i in range(xs.shape[1]):
                preds[f'xs_{sifb.length_scale_name[i]}'] = xs[:, i]

            t2 = time.time()

            # ----
            # store results (parameters) and predictions
            # ----

            ln, lt = EASE2toWGS84_New(x_, y_)

            res = {
                "date": date,
                "x": x_,
                "y": y_,
                "lon": ln,
                "lat": lt,
                "grid_loc_0": grid_loc[0],
                "grid_loc_1": grid_loc[1],
                "num_inputs": len(inputs),
                **opt_hyp,
                "run_time": t1-t0,
                **{f"scale_{sifb.length_scale_name[i]}": si for i, si in enumerate(sifb.scale_inputs)},
                "scale_output": sifb.scale_outputs
            }

            # store in dataframe - for easy writing / appending to file
            rdf = pd.DataFrame(res, index=[i])
            pdf = pd.DataFrame(preds)

            if append_to_file:
                # append results to file
                rdf.to_csv(res_file, mode="a", header=not os.path.exists(res_file),
                           index=False)
                pdf.to_csv(pred_file, mode="a", header=not os.path.exists(res_file),
                           index=False)

            all_res.append(rdf)
            all_preds.append(pdf)

    all_res = pd.concat(all_res)
    all_preds = pd.concat(all_preds)

    # --
    # total run time
    # --

    t_total1 = time.time()
    print(f"total run time: {t_total1 - t_total0:.2f}")