#!/usr/bin/env python
# coding: utf-8

# install required packages

# In[1]:


import sys
sys.version


# In[2]:


get_ipython().system('pip install gpflow==2.5.2')
get_ipython().system('pip install pyproj')
# NOTE: installing cartopy via pip just to avoid import error
# - cartopy should be installed via: conda install -c conda-forge cartopy
# !pip install cartopy


# config / parameters
# 

# In[3]:


branch_name = "sparse_dev"
# directory on google drive where to 
# work_sub_dir = ["MyDrive", "workspace"]


# mount google drive (use to save results)  - requires login

# In[4]:


import subprocess
import json
from google.colab import drive
import os
import sys


gdrive_mount = '/content/gdrive'
# # requires giving access to google drive account
drive.mount(gdrive_mount)


# git pull repository

# In[5]:


import re
# change 'workspace' as needed
# work_dir = os.path.join(gdrive_mount, *["MyDrive", "workspace"])
work_dir = "/content"

# change to working directory
# os.chdir(work_dir)
assert os.path.exists(work_dir), f"workspace directory: {work_dir} does not exist"
os.chdir(work_dir)

# !git clone https://github.com/William-gregory/OptimalInterpolation.git
# url suffix for cloning repp
url = "https://github.com/William-gregory/OptimalInterpolation.git"

# repository directory
repo_dir = os.path.join(work_dir, os.path.basename(url))
repo_dir = re.sub("\.git$", "", repo_dir)

# TODO: put a try except here 
# clone the repo

try:
    git_clone = subprocess.check_output( ["git", "clone", url] , shell=False)
except Exception as e:
    # get non-zero exit status 128: if the repo already exists?
    print(e)

print(f"changing directory to: {repo_dir}")

os.chdir(repo_dir)


# Change branch 

# In[6]:


# --
# change branch - review this
# --

try:
    git_checkout = subprocess.check_output(["git", "checkout", "-t", f"origin/{branch_name}"], shell=False)
    print(git_checkout.decode("utf-8") )
except Exception as e:
    git_checkout = subprocess.check_output(["git", "checkout",  f"{branch_name}"], shell=False)
    print(git_checkout.decode("utf-8") )



# In[7]:


# git pull to ensure have the latest
git_pull = subprocess.check_output(["git", "pull"], shell=False)
print(git_pull.decode("utf-8") )


# In[8]:


# add directory to containing repository to sys.path, so can import as a package
# if repo_dir not in sys.path:
#     # tmp_dir = os.path.dirname(repo_dir)
#     print(f"adding {repo_dir} to sys.path")
#     sys.path.extend([])

if work_dir not in sys.path:
    # tmp_dir = os.path.dirname(repo_dir)
    print(f"adding {work_dir} to sys.path")
    sys.path.extend([work_dir])


# In[9]:


# TODO: only downlaod if it does not already exist
import gdown
import zipfile

# print("there was some sort of issue downloading the entire folder structure")
print("will try downloading the zipped version")
# url = "https://drive.google.com/file/d/1c7h6HTT-wbCq_ZKBYLJSSln4tanlLEMZ"
# id = "1c7h6HTT-wbCq_ZKBYLJSSln4tanlLEMZ"


data_dir = os.path.join(repo_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# https://drive.google.com/file/d/1djlaZ2EKbm9pNAEt3w58WJtBA4NyQsNE/view?usp=sharing
id_zip = [
    # {"id": "1ckoowmCwh4tG76sIxXZuVaSSQ0tv8KTU", "zip": "auxiliary.zip", "dirname": "aux"},
    {"id": "1djlaZ2EKbm9pNAEt3w58WJtBA4NyQsNE", "zip": "new_aux.zip", "dirname": "aux"},
    {"id": "1cIh9lskzmL6C7EYV8lmJJ5YaJgKqOZHT", "zip": "CS2S3_CPOM.zip", "dirname": "CS2S3_CPOM"},
    {"id": "1gXsvtxZcWpBALomgeqn9kcfyCtKD3fkz", "zip": "raw_along_track.zip", "dirname": "RAW"},
    # legacy data
    # {"id": "1HcKZD_F3esIPc2NbWlexvXbVOSAlso9m", "zip": "aux_legacy.zip", "dirname": "aux"},
    # {"id": "1Kekh43yTDVJXfSjUrPDV6ZEXXIJhzdXC", "zip": "CS2S3_CPOM_legacy.zip", "dirname": "CS2S3_CPOM"},
]

# TODO: check if output dir already exists: aux and CS2S3_CPOM
for _ in id_zip:
    id = _['id']
    zip = _['zip']
    dirn = _.get('dirname', "")

    if os.path.exists(os.path.join(data_dir, dirn)):
        # print(f"dir{}")
        continue
    # put data in data dir in repository
    output = os.path.join(data_dir, zip)
    gdown.download(id=id, output=output, use_cookies=False)

    # un zip to path
    print("unzipping")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(path=data_dir)


# In[10]:


import os


# In[11]:


# change to repo_dir so can get git info
os.chdir(repo_dir)

# set output directory
# output_base_dir = os.path.join(gdrive_mount, "MyDrive", "Dissertation")


# In[ ]:



# ------------------------------------------------------------
# ------------------------------------------------------------
# 
#                   SCRIPT STARTS HERE
#
# ------------------------------------------------------------
# ------------------------------------------------------------


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
import tensorflow as tf
import tensorflow_probability as tfp

import time
from OptimalInterpolation import get_data_path, get_path
from OptimalInterpolation.sea_ice_freeboard import SeaIceFreeboard

from OptimalInterpolation.utils import EASE2toWGS84_New, get_git_information, move_to_archive



# In[ ]:





# #  change output_base_dir as needed
# output_base_dir = get_path("results")
output_base_dir = os.path.join(gdrive_mount, "MyDrive", "Dissertation")


# ---
# input config
# ---


config = {
    "dates": ["20181202"],#["20181201"],#, "20190101", "20190201", "20190301"],
    "optimise": True,
    "file_suffix": "_repeat",
    "output_dir": os.path.join(output_base_dir, "refactor_svgp2"),
    "inclusion_radius": 300,
    "days_ahead": 4,
    "days_behind": 4,
    "data_dir": "package",
    "season": "2018-2019",
    "grid_res": 50,
    "coarse_grid_spacing": 1,
    "min_inputs": 5,
    "verbose": 1,
    "engine": "GPflow_svgp",
    "kernel": "Matern32",
    # "prior_mean_method": "demean_outputs",
    # "mean_function": "constant",
    # "hold_out": ["S3B", "S3A", "CS2_SAR", None],
    "hold_out": None,
    "load_params": True,
    "predict_on_hold": True,
    "scale_inputs": True,
    "scale_outputs": False,
    "bound_length_scales": True,
    "append_to_file": True,
    "overwrite": False,
    "post_process": {
        "prev_results_dir": None,
        "prev_results_file": None,
        "clip_and_smooth": True,
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
        "num_inducing_points": 500,
        "min_obs_for_svgp": 5000
        },
    # when not using minbatch can use a low (150?) number for maxiter
    "optimise_params": {
        "use_minibatch": False,
        "gamma": 1.0,
        "learning_rate": 0.07,
        "trainable_inducing_variable": False,
        "minibatch_size": 2000,
        "maxiter": 120,
        "log_freq": 10,
        "persistence": 10,
        "early_stop": True,
        "save_best": True
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
prior_mean_method = config.get("prior_mean_method", "fyi_average")
hold_out = config.get("hold_out", None)

scale_inputs = config.get("scale_inputs", False)
scale_inputs = [1 / (grid_res * 1000), 1 / (grid_res * 1000), 1.0] if scale_inputs else [1.0, 1.0, 1.0]

scale_outputs = config.get("scale_outputs", False)
scale_outputs = 100. if scale_outputs else 1.

append_to_file = config.get("append_to_file", True)
overwrite = config.get("overwrite", True)

# use holdout location as GP location selection criteria (in addition to coarse_grid_spacing, etc)
pred_on_hold_out = config.get("predict_on_hold", True)

bound_length_scales = config.get("bound_length_scales", True)

mean_function = config.get("mean_function", None)
file_suffix = config.get("file_suffix", "")
post_process_config = config.get("post_process", {})

load_params = config.get("load_params", False)


inducing_point_params = config.get("inducing_point_params", {})

optimise_params = config.get("optimise_params", {})

# -----
# initialise SeaIceFreeboard object
# -----

sifb = SeaIceFreeboard(grid_res=f"{grid_res}km",
                       length_scale_name=["x", "y", "t"],
                       verbose=3,
                       rng_seed=1234)

# ---
# read / load data
# ---

if data_dir == "package":
    data_dir = get_data_path()

assert os.path.exists(data_dir)

sifb.load_data(aux_data_dir=os.path.join(data_dir, "aux"),
                sat_data_dir=os.path.join(data_dir, "CS2S3_CPOM"),
                raw_data_dir=os.path.join(data_dir, "RAW"),
                season=season)


# increment over hold_out (e.g. don't combine)
if hold_out is None:
    hold_outs = [None]
else:
    if isinstance(hold_out, str):
        hold_outs = [hold_out]
    else:
        hold_outs = hold_out

for hold_out in hold_outs:
    print("-" * 100)
    print(f"hold_out: {hold_out}")
    print("-" * 10)

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
                 prior_mean_method=prior_mean_method,
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
                 print_every=50,
                 inducing_point_params=inducing_point_params,
                 optimise_params=optimise_params)


# In[ ]:





# 

# 

# In[ ]:


# new_res_dir = os.path.join(gdrive, "refactored")
# os.makedirs(new_res_dir)


# In[ ]:


# chk = subprocess.check_output(['cp', '-r', results_dir, gdrive], shell=False)


# In[ ]:


# chk


# In[ ]:




