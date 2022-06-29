#!/usr/bin/env python
# coding: utf-8

# # install required packages
#
# # In[17]:
#
#
# import sys
# sys.version
#
#
# # In[18]:
#
#
# get_ipython().system('pip install gpflow')
# get_ipython().system('pip install pyproj')
#
#
# # config / parameters
# #
#
# # In[19]:
#
#
# branch_name = "sparse_dev"
# # directory on google drive where to
# # work_sub_dir = ["MyDrive", "workspace"]
#
#
# # mount google drive (use to save results)  - requires login
#
# # In[20]:
#
#
# import subprocess
# import json
# from google.colab import drive
# import os
# import sys
#
#
# gdrive_mount = '/content/gdrive'
# # # requires giving access to google drive account
# drive.mount(gdrive_mount)
#
#
# # git pull repository
#
# # In[21]:
#
#
# import re
# # change 'workspace' as needed
# # work_dir = os.path.join(gdrive_mount, *["MyDrive", "workspace"])
# work_dir = "/content"
#
# # change to working directory
# # os.chdir(work_dir)
# assert os.path.exists(work_dir), f"workspace directory: {work_dir} does not exist"
# os.chdir(work_dir)
#
# # !git clone https://github.com/William-gregory/OptimalInterpolation.git
# # url suffix for cloning repp
# url = "https://github.com/William-gregory/OptimalInterpolation.git"
#
# # repository directory
# repo_dir = os.path.join(work_dir, os.path.basename(url))
# repo_dir = re.sub("\.git$", "", repo_dir)
#
# # TODO: put a try except here
# # clone the repo
#
# try:
#     git_clone = subprocess.check_output( ["git", "clone", url] , shell=False)
# except Exception as e:
#     # get non-zero exit status 128: if the repo already exists?
#     print(e)
#
# print(f"changing directory to: {repo_dir}")
#
# os.chdir(repo_dir)
#
#
# # Change branch
#
# # In[22]:
#
#
# # --
# # change branch - review this
# # --
#
# try:
#     git_checkout = subprocess.check_output(["git", "checkout", "-t", f"origin/{branch_name}"], shell=False)
#     print(git_checkout.decode("utf-8") )
# except Exception as e:
#     git_checkout = subprocess.check_output(["git", "checkout",  f"{branch_name}"], shell=False)
#     print(git_checkout.decode("utf-8") )
#
#
#
# # In[23]:
#
#
# # git pull to ensure have the latest
# git_pull = subprocess.check_output(["git", "pull"], shell=False)
# print(git_pull.decode("utf-8") )
#
#
# # In[24]:
#
#
# # add directory to containing repository to sys.path, so can import as a package
# # if repo_dir not in sys.path:
# #     # tmp_dir = os.path.dirname(repo_dir)
# #     print(f"adding {repo_dir} to sys.path")
# #     sys.path.extend([])
#
# if work_dir not in sys.path:
#     # tmp_dir = os.path.dirname(repo_dir)
#     print(f"adding {work_dir} to sys.path")
#     sys.path.extend([work_dir])
#
#
# # In[25]:
#
#
# # TODO: only downlaod if it does not already exist
# import gdown
# import zipfile
#
# # print("there was some sort of issue downloading the entire folder structure")
# print("will try downloading the zipped version")
# # url = "https://drive.google.com/file/d/1c7h6HTT-wbCq_ZKBYLJSSln4tanlLEMZ"
# # id = "1c7h6HTT-wbCq_ZKBYLJSSln4tanlLEMZ"
#
#
# data_dir = os.path.join(repo_dir, "data")
# os.makedirs(data_dir, exist_ok=True)
#
# id_zip = [
#     {"id": "1ckoowmCwh4tG76sIxXZuVaSSQ0tv8KTU", "zip": "auxiliary.zip"},
#     {"id": "1cIh9lskzmL6C7EYV8lmJJ5YaJgKqOZHT", "zip": "CS2S3_CPOM.zip"},
# ]
#
# for _ in id_zip:
#     id = _['id']
#     zip = _['zip']
#     # put data in data dir in repository
#     output = os.path.join(data_dir, zip)
#     gdown.download(id=id, output=output, use_cookies=False)
#
#     # un zip to path
#     print("unzipping")
#     with zipfile.ZipFile(output, 'r') as zip_ref:
#         zip_ref.extractall(path=data_dir)
#
#
# # In[26]:
#
#
# # import os
#
# # import os
# # os.listdir(work_dir)
# # os.listdir(data_dir)
#
# # # REMOVE THIS
# # print("confirming folder structure of content")
# # for root, subs, files in os.walk(repo_dir, topdown=True):
# #     print("-"*50)
# #     print(f"root: {root}")
# #     print(f"sub dirs: {subs}")
# #     print(f"files: {files}")
#
#
# # In[27]:
#
#
# # change to repo_dir so can get git info
# os.chdir(repo_dir)

# set output directory
# output_base_dir = os.path.join(gdrive_mount, "MyDrive", "Dissertation")

from OptimalInterpolation import get_data_path, get_path
output_base_dir = get_data_path("results")

# In[28]:


# calculate the hyper-parameters for GP on freeboard cover using a config
# - date(s)
# - window size
# - radius of inclusion
# - freeboard season

import warnings
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
from OptimalInterpolation import get_data_path, get_path
from OptimalInterpolation.utils import grid_proj, get_git_information, load_data, split_data2, move_to_archive


# In[29]:



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
    "dates": ["20181201", "20190101", "20190201", "20190301"], # "20181201"
    "inclusion_radius": 300,
    "days_ahead": 4,
    "days_behind": 4,
    "data_dir": "package",
    "season": "2018-2019",
    "grid_res": 50,
    "coarse_grid_spacing": 4,
    "min_inputs": 10,
    "verbose": 1,
    "initialise_with_neighbors": False,
    # "hold_out": ["S3B"],
    # "predict_on_hold": True,
    "scale_inputs": True,
    "output_dir": os.path.join(output_base_dir, "paper_prior_mean_scale_inputs")
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

# initialise hyper parameters with neighbours values, if they exist
init_w_neigh = config.get("initialise_with_neighbors", False)

# -
# hold out data: used for cross validation
hold_out = config.get("hold_out", [])

# scale the x,y dimension of the inputs?
scale_inputs = config.get("scale_inputs", False)

scale_dict = {
    "x": grid_res * 1000 if scale_inputs else 1.0,
    "y": grid_res * 1000 if scale_inputs else 1.0,
    "t": 1.0
}

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


# In[30]:


t_total0 = time.time()

from OptimalInterpolation.data_loader import DataLoader

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


# this contains a (x,y,t) numpy array of only first-year-ice freeboards.
# We use this to define the prior mean
# cs2_FYI = np.load(
#     datapath + '/CS2_25km_FYI_20181101-20190428.npy')

# # HARDCODED: drop the first 25 days to align with obs data
# # TODO: this should be done in a more systematic way
# cs2_FYI = cs2_FYI[..., 25:]

# TODO: make sure dates are aligned to satellite data, read in same way
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


# In[31]:


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


# get the datetime of the run
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

for date in calc_dates:

    date_dir = os.path.join(output_dir, date)
    if not os.path.exists(date_dir):
        print(f"date_dir: {date_dir}\n does not exists, creating")
        os.makedirs(date_dir)

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
    # mean = np.nanmean(cs2_FYI[..., (cday - days_behind):(cday + days_ahead + 1)]).round(4)
    mean = np.nanmean(cs2_FYI[..., (cday - (days_behind + days_ahead + 1)):cday]).round(4)

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

    # store hyper parameters in a dict for each grid location
    # - this is so can initialise hyper parameters with neighbours
    # - if  init_w_neigh is True 
    hp_dict = {}
    

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
            # print(f"for (x,y)= ({x_:.2f}, {y_:.2f})\nthere were only {len(ID)} < {min_inputs} points, skipping")
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

        # scale inputs - redundant if scale_inputs=False
        inputs = inputs / np.array([scale_dict['x'], scale_dict['y'], scale_dict['t']])

        # ----
        # GPflow
        # ----
        t0 = time.time()


        # ---
        # initialise hyper parameters with neighbours points?
        # ---

        length_scales = [grid_res * 1000 / scale_dict['x'],
                         grid_res * 1000 / scale_dict['y'],
                         1.0 / scale_dict['t']]
        kernel_var = 1.0
        noise_var = 1.0 


        used_n_hp = False 
        if init_w_neigh:
            params = []
            # TODO: consider different ranges here, allow as input parameter
            neigh_range = range(-coarse_grid_spacing, coarse_grid_spacing+1)
            for ii in neigh_range:
                for jj in neigh_range:          
                    # location of neighbour
                    n_loc = (grid_loc[0] + ii, grid_loc[1] + jj)
                    # if the location already has hyper parameters get those 
                    # - nested dict
                    if n_loc in hp_dict:
                        params.append(hp_dict[n_loc])
            param_count = len(params)
            if param_count:
                # print("using neighbors hyper parameter values")
                # print("params")
                # print(params)
                # print(params[0])
                ave_param = {hp: sum([p[hp] for p in params]) / param_count
                             for hp in ["ls_y", "ls_x", "ls_t", "kernel_variance", "likelihood_variance"]}
                length_scales = [ave_param['ls_x'], ave_param['ls_y'], ave_param['ls_t'] ]
                kernel_var = ave_param['kernel_variance']
                noise_var = ave_param['likelihood_variance']

                # these values can be 
                if np.isnan(noise_var) | np.isinf(noise_var):
                    print(f"noise_var: {noise_var}")
                    noise_var = 1.0

                if np.isnan(kernel_var) | np.isinf(kernel_var):
                    print(f"kernel_var: {kernel_var}")
                    kernel_var = 1.0

                used_n_hp = True

        # noise_variance can't be too small 
        noise_var = 1e-5 if noise_var <= 1e-6 else noise_var
                                 

        # kernel
        k = gpflow.kernels.Matern32(lengthscales=length_scales,
                                    variance=kernel_var)

        # GPR object
        m = gpflow.models.GPR(data=(inputs, (outputs - mX)[:, None]),
                              kernel=k, mean_function=None,
                              noise_variance=noise_var)

        # solve for optimal (max log like) hyper parameters
        opt = gpflow.optimizers.Scipy()

        # %%
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=10000))
        # print_summary(m)

        if not opt_logs['success']:
            print("*" * 10)
            print("optimization failed! will skip")
            print(f"used_n_hp: {used_n_hp}, kernel_var: {kernel_var}, noise_var: {noise_var}")
            print(f"grid_loc: {grid_loc}")
            tmp = pd.DataFrame({"grid_loc_0": grid_loc[0],
                    "grid_loc_1": grid_loc[1],
                    "reason": f"did not converge"},
                    index=[i])

            tmp.to_csv(skip_file, mode='a',
                        header=not os.path.exists(skip_file),
                        index=False)
            continue
            # length_scales = [grid_res * 1000, grid_res * 1000, 1.0]
            # kernel_var = 1.0
            # noise_var = 1.0

            # # kernel
            # k = gpflow.kernels.Matern32(lengthscales=length_scales,
            #                             variance=kernel_var)
            # # GPR object
            # m = gpflow.models.GPR(data=(inputs, (outputs - mX)[:, None]),
            #                         kernel=k, mean_function=None,
            #                         noise_variance=noise_var)
            # # solve for optimal (max log like) hyper parameters
            # opt = gpflow.optimizers.Scipy()
            # # %%
            # opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=10000))

            # print(f"opt_logs['success']: {opt_logs['success']}")
            # print(f"grid_loc: {grid_loc}")


        t1 = time.time()

        # get the run time
        run_time = t1 - t0

        # extract the hyper parameters
        gpf_hyp = np.concatenate([m.kernel.lengthscales.numpy(),
                                    np.atleast_1d(m.kernel.variance.numpy()),
                                    np.atleast_1d(m.likelihood.variance.numpy())])

        # TODO: double check to see if these are the right way around 
        #  - this the first dimension not y?
        # scale the length scales backup
        lscale = {nn: m.kernel.lengthscales.numpy()[_] * scale_dict[re.sub("ls_", "", nn)]
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
            "used_n_hp": used_n_hp,
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
        
        # store hyper parameters in dict - using grid_loc as key
        if init_w_neigh:
            hp_dict[( grid_loc[0],  grid_loc[1])] = {i: res[i] 
                                                     for i in ['ls_x', 'ls_y', 'ls_t', 'kernel_variance', 'likelihood_variance']}

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

        x_s = xFB[n_select_loc][:, None] / scale_dict['x']
        y_s = yFB[n_select_loc][:, None] / scale_dict['y']
        lon_s = lonFB[n_select_loc][:, None]
        lat_s = latFB[n_select_loc][:, None]
        t_s = np.ones(len(x_s))[:, None] * days_behind / scale_dict['t']
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


noise_var


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




