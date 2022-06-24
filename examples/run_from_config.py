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
import warnings
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



if __name__ == "__main__":

    # ***** CHANGE THIS ON COLAB ******
    output_base_dir = get_path("results")
    print(f"output_base_dir: {output_base_dir}")


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
        "dates": ["20181201"],
        "inclusion_radius": 300,
        "days_ahead": 4,
        "days_behind": 4,
        "data_dir": "package",
        "season": "2018-2019",
        "grid_res": 25,
        "coarse_grid_spacing": 1,
        "min_inputs": 10,
        "verbose": 1,
        "hold_out": ["S3A"],
        "predict_on_hold": True,
        "output_dir": os.path.join(output_base_dir, "ease_projection")
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
    # CURRENTLY ONLY ALLOW 2018-2019
    assert season == "2018-2019"

    grid_res = config.get("grid_res", 25)
    # min sea ice cover - when loading data set sie to nan if < min_sie
    min_sie = config.get("min_sie", 0.15)

    # spacing for coarse grid - let be > 1 to select subset of points
    coarse_grid_spacing = config.get("coarse_grid_spacing", 1)

    # min number of inputs to calibrate GP on
    min_inputs = config.get("min_inputs", 10)

    # -
    # hold out data: used for cross validation
    hold_out = config.get("hold_out", [])

    # predict only on hold out locations
    pred_on_hold_out = config.get("predict_on_hold", False)

    if len(hold_out):
        print(f"will hold_out data from:\n{hold_out}\n(from prediction date)")
        print(f"pred_on_hold_out (predict only on hold out points) = {pred_on_hold_out}")

    output_dir = config["output_dir"]


    # make an output dir based on parameters
    # - recall date subdirectories will be added
    # holdouts = ""
    tmp_dir = f"radius{incl_rad}_daysahead{days_ahead}_daysbehind{days_behind}_gridres{grid_res}_season{season}_coarsegrid{coarse_grid_spacing}_holdout{'|'.join(hold_out)}"
    output_dir = os.path.join(output_dir, tmp_dir)

    os.makedirs(output_dir, exist_ok=True)
    # assert os.path.exists(output_dir), f"output_dir: {output_dir} \n does not exists, expect it to"


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

    # ---
    # write config to file
    # --

    # TODO: put this somewhere else - to avoid over writing

    with open(os.path.join(output_dir, "input_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # In[ ]:

    t_total0 = time.time()

    # ----
    # load data
    # ----
    print("loading data")
    # obs, sie, dates, xFB, yFB, lat, lon = load_data(datapath, grid_res, season,
    #                                                 dates_to_datetime=False,
    #                                                 trim_xy=1, min_sie=min_sie)

    # create a DataLoader object
    dl = DataLoader(grid_res=f"{grid_res}km", seasons=[season], verbose=2)

    # load aux(iliary) data
    dl.load_aux_data(aux_data_dir=get_data_path("aux"),
                     season=season)
    # load obs(servation) data
    dl.load_obs_data(sat_data_dir=get_data_path("CS2S3_CPOM"),
                     grid_res=f"{grid_res}km")

    # check hold_out values are valid (satellite names)
    for ho in hold_out:
        assert ho in dl.obs['dims']['sat'], f"hold_out sat: {ho} not in obs dims:\n{dl.obs['dims']['sat']}"

    # --
    # data for prior mean - review
    # --

    # this contains a (x,y,t) numpy array of only first-year-ice freeboards.
    # We use this to define the prior mean
    # cs2_FYI = np.load(
    #     datapath + '/CS2_25km_FYI_20181101-20190428.npy')

    # # HARDCODED: drop the first 25 days to align with obs data
    # # TODO: this should be done in a more systematic way
    # cs2_FYI = cs2_FYI[..., 25:]

    cs2_FYI = np.load(
        datapath + f'/aux/CS2_{grid_res}km_FYI_20181101-20190428.npy')
    # create an array of dates
    cs2_FYI_dates = np.arange(np.datetime64("2018-11-01"), np.datetime64("2019-04-29"))
    cs2_FYI_dates = np.array([re.sub("-", "", i) for i in cs2_FYI_dates.astype(str)])

    # trimming to align with data
    xFB = dl.aux['x'][:-1, :-1]
    yFB = dl.aux['y'][:-1, :-1]
    lonFB = dl.aux['lon'][:-1, :-1]
    latFB = dl.aux['lat'][:-1, :-1]

    # In[ ]:

    # NOTE: the t values are integers in window, not related to actual dates
    # x_train, y_train, t_train, z = data_select(date, dates, obs, xFB, yFB,
    #                                         days_ahead=days_ahead,
    #                                         days_behind=days_behind)
    # dl.obs['data'].shape
    # xFB.shape

    # In[ ]:

    # ----
    # for each date calculate hyper-parameters, make predictions, store values
    # ----

    # make a coarse grid - can be used to select a subset of points
    cgrid = dl.coarse_grid(coarse_grid_spacing,
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

        # ---
        # write config to file - will end up doing this for each date
        # ---

        with open(os.path.join(date_dir, "input_config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # results will be written to file
        res_file = os.path.join(date_dir, "results.csv")
        pred_file = os.path.join(date_dir, "prediction.csv")
        # bad results will be written to :
        skip_file = os.path.join(date_dir, "skipped.csv")

        # --
        # hold out data
        # --

        # let hold_out_bool flag up where the hold out locations are
        # - for the current date
        hold_out_bool = np.zeros(dl.obs['data'].shape[:2], dtype=bool)

        if len(hold_out):
            print("some hold out data provided")
            print(hold_out)
            # copy observation data
            # - so can set hold_out data to np.nan
            obs = dl.obs['data'].copy()

            for ho in hold_out:
                print(f"removing: {ho} data")
                # get the location of the hold_out (sat)
                sat_loc = np.in1d(dl.obs['dims']['sat'], ho)
                date_loc = np.in1d(dl.obs['dims']['date'], date)
                # get hold_out data observations locations
                hold_out_bool[~np.isnan(obs[:, :, sat_loc, date_loc][..., 0])] = True
                # set the observations at the hold out location to nan
                obs[:, :, sat_loc, date_loc] = np.nan

        # select x,y,t and z (freeboard) data for date
        # NOTE: the t values are integers in window, not related to actual dates
        x_train, y_train, t_train, z = dl.data_select(date, dates, obs, xFB, yFB,
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

        # mean = np.nanmean(cs2_FYI[..., (day - days_behind):(day + days_ahead + 1)]).round(4)
        # TODO: should have checks that range is valid here
        print("using CS2_FYI data for prior mean - needs review")
        cday = np.where(np.in1d(cs2_FYI_dates, date))[0][0]
        # TODO: should this be trailing 31 days?
        mean = np.nanmean(cs2_FYI[..., (cday - days_behind):(cday + days_ahead + 1)]).round(4)

        # ---
        # select locations with sea ice cover exists to predict on
        # ---

        # select bool will determine which locations are predicted for
        select_bool = ~np.isnan(sie[..., day]) & cgrid

        # if predicting only on the hold out locations, include those in select_bool
        if pred_on_hold_out:
            print("will predict only on hold_out data locations (non nan)")
            # require there are at least some positions to predict on
            assert hold_out_bool.any(), f"pred_on_hold_out: {pred_on_hold_out}\nhowever hold_out_bool.any(): {hold_out_bool.any()}"

            select_bool = select_bool & hold_out_bool

        if not select_bool.any():
            warnings.warn("there are no points to predict on, will do nothing, check configuration")

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

            # TODO: move this above
            # - this does not work as expect - use the long, lat data from aux
            # mplot = grid_proj(llcrnrlon=-90, llcrnrlat=75, urcrnrlon=-152, urcrnrlat=82)
            # ln, lt = mplot(x_, y_, inverse=True)

            # getting the pre-calculated lon, lat data
            ln = lonFB[grid_loc]
            lt = latFB[grid_loc]

            # get the points from the input data within radius
            ID = X_tree.query_ball_point(x=[x_, y_],
                                         r=incl_rad * 1000)

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

            t0 = time.time()

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
            lon_s = lonFB[n_select_loc][:, None]
            lat_s = latFB[n_select_loc][:, None]
            t_s = np.ones(len(x_s))[:, None] * days_behind
            xs = np.concatenate([x_s, y_s, t_s], axis=1)

            y_pred = m.predict_y(Xnew=xs)
            f_pred = m.predict_f(Xnew=xs)

            t1 = time.time()

            pred = {
                "grid_loc_0": grid_loc[0],
                "grid_loc_1": grid_loc[1],
                "proj_loc_0": n_select_loc[0],
                "proj_loc_1": n_select_loc[1],
                "x": x_s[:, 0],
                "y": y_s[:, 0],
                "lon": lon_s[:, 0],
                "lat": lat_s[:, 0],
                "f*": f_pred[0].numpy()[:, 0],
                "f*_var": f_pred[1].numpy()[:, 0],
                "y_var": y_pred[1].numpy()[:, 0],
                "mean": mean,
                "run_time": t1 - t0
            }

            tmp = pd.DataFrame(pred)

            # append results to file
            tmp.to_csv(pred_file, mode="a", header=not os.path.exists(pred_file),
                       index=False)

    # In[ ]:

    t_total1 = time.time()
    print(f"total run time: {t_total1 - t_total0:.2f}")

    with open(os.path.join(output_dir, "total_runtime.txt"), "+w") as f:
        f.write(f"runtime: {t_total1 - t_total0:.2f} seconds")

    # In[ ]:

    # In[ ]:

    # # from will
    # import cartopy.crs as ccrs
    # import cartopy.feature as cfeat

    # fig, ax = plt.subplots(1, figsize=(5, 5),
    #                        subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
    # ax.coastlines(resolution='50m', color='white')
    # ax.add_feature(cfeat.LAKES, color='white', alpha=.5)
    # # ax.add_feature(cfeat.RIVERS, color='white', alpha=.1)
    # ax.add_feature(cfeat.LAND, color=(0.8, 0.8, 0.8))

    # ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())  # lon_min,lon_max,lat_min,lat_max

    # plt.show()

    # In[ ]:




