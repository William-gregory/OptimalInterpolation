# calculate the hyper-parameters for GP on freeboard cover using a config
# - date(s)
# - window size
# - radius of inclusion
# - freeboard season

import json
import os
import re
import datetime
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform, pdist, cdist
import scipy.optimize
import gpflow
from gpflow.utilities import print_summary

import time
from OptimalInterpolation.data_loader import DataLoader
from OptimalInterpolation import get_data_path, get_path
from OptimalInterpolation.utils import grid_proj, get_git_information, load_data, split_data2


def data_select(date, dates, obs, xFB, yFB, days_ahead=4, days_behind=4):
    # for a given date select
    print("selecting data")

    # find day location based on given date value
    day = np.where(np.in1d(dates, date))[0][0]

    # check the days about the window
    assert (day-days_behind) >= 0, \
        f"date:{date}, days_behind:{days_behind} gives day-days_behind: {day-days_behind}, which must be >= 0"

    assert (day+days_ahead) <= (len(dates) - 1), \
        f"date:{date}, days_ahead:{days_ahead} gives day+days_behind= {day+days_ahead}, " \
        f"which must be <= (len(dates) - 1) = {len(dates) - 1}"

    # the T days of training data from all satellites
    sat = obs[:, :, :, (day-days_behind):(day+days_ahead+1)]

    # select data
    x_train, y_train, t_train, z = split_data2(sat, xFB, yFB)

    return x_train, y_train, t_train, z


def coarse_grid(grid_space, grid_space_offset=0, x_size=320, y_size=320):
    # create a 2D array of False except along every grid_space points
    # - can be used to select a subset of points from a grid
    # NOTE: first dimension is treated as y dim
    cb_y, cb_x = np.zeros(x_size, dtype='bool'), np.zeros(y_size, dtype='bool')
    cb_y[(np.arange(len(cb_y)) % grid_space) == grid_space_offset] = True
    cb_x[(np.arange(len(cb_x)) % grid_space) == grid_space_offset] = True
    return cb_y[:, None] * cb_x[None, :]


if __name__ == "__main__":

    # TODO: let config be an input (via sys.argv)
    # TODO: have an option to calculate using the pure python approach
    # TODO: allow for prior mean to be determined in a more robust way
    # TODO: restrict observations to correspond to points where sie cover >= 0.15
    # TODO: allow for kernel to be specified
    # TODO: write bad results to file
    # TODO: allow for over writing, default should be to no allow
    # TODO: need to be in correct directory to get git info
    # TODO: consider storing results in json file, one per grid point, can get more info

    # ---
    # configuration (defined inline here)
    # ---

    config = {
        "dates": ["20181223"],
        "inclusion_radius": 300,
        "days_ahead": 4,
        "days_behind": 4,
        "data_dir": "package",
        "season": "2018-2019",
        "grid_res": 25,
        "coarse_grid_spacing": 4,
        "min_inputs": 10,
        "verbose": 1,
        "output_dir": get_path("results")
    }

    print("using config:")
    print(json.dumps(config, indent=4))

    # ----
    # parameters: extract from config
    # ----

    # dates to calculate hyper parameters
    calc_dates = config['dates']
    calc_dates = [calc_dates] if not isinstance(calc_dates, list) else calc_dates

    # radius about a location to include points - in km
    incl_rad = config.get("inclusion_radius", 300)

    # directory containing freeboard data, if 'package' is given will use 'data' directory in package
    datapath = config.get("data_dir", "package")
    if datapath == "package":
        datapath = get_data_path()

    # days ahead and behind given date to include for inputs
    days_ahead = config.get("days_ahead", 4)
    days_behind = config.get("days_behind", 4)

    season = config.get("season", "2018-2019")
    grid_res = config.get("grid_res", 25)
    # min sea ice cover - when loading data set sie to nan if < min_sie
    min_sie = config.get("min_sie", 0.15)

    # spacing for coarse grid - let be > 1 to select subset of points
    coarse_grid_spacing = config.get("coarse_grid_spacing", 1)

    # min number of inputs to calibrate GP on
    min_inputs = config.get("min_inputs", 10)

    output_dir = config["output_dir"]
    assert os.path.exists(output_dir), f"output_dir: {output_dir} \n does not exists, expect it to"

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

    # ---
    # write config to file
    # --

    config["run_info"] = run_info
    with open(os.path.join(output_dir, "input_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # ----
    # load data
    # ----

    # create a DataLoader object
    dl = DataLoader()

    # load aux(iliary) data
    dl.load_aux_data(aux_data_dir=get_data_path("aux"),
                     season=season)
    # load obs(servation) data
    dl.load_obs_data(sat_data_dir=get_data_path("CS2S3_CPOM"))

    # print("loading data")
    # obs, sie, dates, xFB, yFB, lat, lon = load_data(datapath, grid_res, season,
    #                                                 dates_to_datetime=False,
    #                                                 trim_xy=1, min_sie=min_sie)

    # this contains a (x,y,t) numpy array of only first-year-ice freeboards.
    # We use this to define the prior mean
    cs2_FYI = np.load(
        datapath + '/CS2_25km_FYI_20181101-20190428.npy')
    # create an array of dates
    cs2_FYI_dates = np.arange(np.datetime64("2018-11-01"), np.datetime64("2019-04-29"))
    cs2_FYI_dates = np.array([re.sub("-", "", i) for i in cs2_FYI_dates.astype(str)])


    # NOTE:

    # HARDCODED: drop the first 25 days to align with obs data
    # TODO: this should be done in a more systematic way
    # cs2_FYI = cs2_FYI[..., 25:]

    # TODO: these points represent the bin edges, should change to be centers
    # trimming to align with data
    xFB = dl.aux['x'][:-1, :-1]
    yFB = dl.aux['y'][:-1, :-1]
    lonFB = dl.aux['lon'][:-1, :-1]
    latFB = dl.aux['lat'][:-1, :-1]

    # ----
    # for each date calculate hyper-parameters, make predictions, store values
    # ----

    # make a coarse grid - can be used to select a subset of points
    cgrid = coarse_grid(coarse_grid_spacing,
                        grid_space_offset=0,
                        x_size=xFB.shape[1],
                        y_size=yFB.shape[0])
    # --
    # extract data needed for training
    # --

    # get the dates from the data
    dates = dl.obs['dims']['date']
    # observation data
    obs = dl.obs['data']
    # sea ice extent
    sie = dl.sie['data']


    # bool array used for projecting onto neighbour
    # - used because easy to select from sie_day at the same time
    n_select = np.zeros(cgrid.shape, dtype='bool')

    for date in calc_dates:

        date_dir = os.path.join(output_dir, date)
        if not os.path.exists(date_dir):
            print(f"date_dir: {date_dir}\n does not exists, creating")
            os.makedirs(date_dir)

        # results will be written to file
        res_file = os.path.join(date_dir, "results.csv")
        pred_file = os.path.join(date_dir, "prediction.csv")
        # bad results will be written to :
        skip_file = os.path.join(date_dir, "skipped.csv")

        # select x,y,t and z (freeboard) data for date
        # NOTE: the t values are integers in window, not related to actual dates
        x_train, y_train, t_train, z = data_select(date, dates, obs, xFB, yFB,
                                                   days_ahead=days_ahead,
                                                   days_behind=days_behind)

        # combine xy data - used for KDtree
        xy_train = np.array([x_train, y_train]).T
        # make a KD tree for selecting point
        X_tree = scipy.spatial.cKDTree(xy_train)

        # get the day - which is in the middle of the data, not at start
        # - it's location corresponds to date
        day = np.where(np.in1d(dates, date))[0][0]

        # --
        # prior mean
        # ---

        # TODO: should have checks that range is valid here
        cday = np.where(np.in1d(cs2_FYI_dates, date))[0][0]
        mean = np.nanmean(cs2_FYI[..., (cday - days_behind):(cday + days_ahead + 1)]).round(4)

        # ---
        # select locations with sea ice cover exists to predict on
        # ---

        select_bool = ~np.isnan(sie[..., day]) & cgrid

        # get the x, y locations
        # x_loc, y_loc = xFB[select_bool], yFB[select_bool]
        num_loc = select_bool.sum()
        select_loc = np.where(select_bool)

        # for each location
        for i in range(num_loc):

            if (i % 100) == 0:
                print("*" * 75)
                print(f"{i + 1}/{num_loc + 1}")

            # select locations
            grid_loc = select_loc[0][i], select_loc[1][i]
            x_ = xFB[grid_loc]
            y_ = yFB[grid_loc]

            # # TODO: move this above
            # mplot = grid_proj(llcrnrlon=-90, llcrnrlat=75, urcrnrlon=-152, urcrnrlat=82)
            # ln, lt = mplot(x_, y_, inverse=True)

            # get the points from the input data within radius
            ID = X_tree.query_ball_point(x=[x_, y_],
                                         r=incl_rad * 1000)

            # if there were too few points then skip
            if len(ID) < min_inputs:
                print(f"for (x,y)= ({x_:.2f}, {y_:.2f})\nthere were only {len(ID)} < {min_inputs} points, skipping")
                tmp = pd.DataFrame({"grid_loc_0": grid_loc[0],
                                    "grid_loc_1": grid_loc[1],
                                    "reason": f"had only {len(ID)} inputs"},
                                   index=[i])

                tmp.to_csv(skip_file, mode='a',
                           header=not os.path.exists(skip_file),
                           index=False)
                continue

            # select points

            inputs = np.array([x_train[ID], y_train[ID], t_train[ID]]).T  # training inputs
            outputs = z[ID]  # training outputs
            n = len(outputs)
            mX = np.full(n, mean)

            # ----
            # GPflow
            # ----
            t0 = time.time()
            # kernel
            k = gpflow.kernels.Matern52(lengthscales=[grid_res * 1000, grid_res * 1000, 1])

            # GPR object
            m = gpflow.models.GPR(data=(inputs, (outputs - mX)[:, None]),
                                  kernel=k, mean_function=None)

            # solve for optimal (max log like) hyper parameters
            opt = gpflow.optimizers.Scipy()

            # %%
            opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=10000))
            # print_summary(m)

            t1 = time.time()

            # get the run time
            run_time = t1 - t0

            # extract the hyper parameters
            gpf_hyp = np.concatenate([m.kernel.lengthscales.numpy(),
                                      np.atleast_1d(m.kernel.variance.numpy()),
                                      np.atleast_1d(m.likelihood.variance.numpy())])


            lscale = {nn: m.kernel.lengthscales.numpy()[_]
                      for _, nn in enumerate(["ls_x", "ls_y", "ls_t"])}
            res = {
                "date": date,
                "x": x_,
                "y": y_,
                "lon": ln,
                "lat": lt,
                "grid_loc_0": grid_loc[0],
                "grid_loc_1": grid_loc[1],
                "num_inputs": len(ID),
                # "lengthscales": m.kernel.lengthscales.numpy(),
                **lscale,
                "kernel_variance": float(m.kernel.variance.numpy()),
                "likelihood_variance": float(m.likelihood.variance.numpy()),
                "run_time": run_time,
                "loglike": m.log_marginal_likelihood().numpy(),
                "mean": mean
            }

            tmp = pd.DataFrame(res, index=[i])

            # append results to file
            tmp.to_csv(res_file, mode="a", header=not os.path.exists(res_file),
                       index=False)


            # ---
            # project to points near by, based on grid spacing
            # ---

            # extract parameters / projection
            # - select neighbouring points
            # TODO: see if there is a neater way of doing this,

            g0, g1 = grid_loc
            l0 = np.max([0, g0 - coarse_grid_spacing])
            u0 = np.min([n_select.shape[0], g0 + coarse_grid_spacing + 1])
            l1 = np.max([0, g1 - coarse_grid_spacing])
            u1 = np.min([n_select.shape[1], g1 + coarse_grid_spacing + 1])

            # select neighbouring points
            n_select[l0:u0, l1:u1] = True
            n_bool = ~np.isnan(sie[..., day]) & n_select
            # set points back
            n_select[l0:u0, l1:u1] = False
            n_select_loc = np.where(n_bool)

            x_s = xFB[n_select_loc][:, None]
            y_s = yFB[n_select_loc][:, None]
            t_s = np.ones(len(x_s))[:, None] * days_behind
            xs = np.concatenate([x_s, y_s, t_s], axis=1)

            y_pred = m.predict_y(Xnew=xs)
            f_pred = m.predict_f(Xnew=xs)

            pred = {
                "grid_loc_0": grid_loc[0],
                "grid_loc_1": grid_loc[1],
                "proj_loc_0": n_select_loc[0],
                "proj_loc_1": n_select_loc[1],
                "x": x_s[:, 0],
                "y": y_s[:, 0],
                "f*": f_pred[0].numpy()[:,0],
                "f*_var": f_pred[1].numpy()[:,0],
                "y_var": y_pred[1].numpy()[:,0],
                "mean": mean
            }

            tmp = pd.DataFrame(pred)

            # append results to file
            tmp.to_csv(pred_file, mode="a", header=not os.path.exists(pred_file),
                       index=False)




            # print(f"num inputs: {len(ID)}, run_time: {run_time}")