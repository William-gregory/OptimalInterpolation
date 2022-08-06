# various helper functions

import numpy as np
from numpy.linalg import multi_dot as mdot
from scipy.spatial.distance import squareform, pdist, cdist
from pyproj import Transformer

import pickle
import pyproj as proj

from datetime import datetime as dt
import datetime
import subprocess
import re
import os
import shutil
import numba as nb

def grid_proj(lon_0=0, boundinglat=60, llcrnrlon=False,
              llcrnrlat=False, urcrnrlon=False, urcrnrlat=False):

    if any([llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat]):
        p = proj.Proj('+proj=stere +R=6370997.0 +units=m +lon_0=' + str(float(lon_0)) + ' +lat_0=90.0',
                      preserve_units=True)
        llcrnrx, llcrnry = p(llcrnrlon, llcrnrlat)
        p = proj.Proj('+proj=stere +R=6370997.0 +units=m +lon_0=' + str(float(lon_0)) + ' +lat_0=90.0 +x_0=' + str(
            -llcrnrx) + ' +y_0=' + str(-llcrnry), preserve_units=True)
    else:
        p = proj.Proj('+proj=stere +R=6370997.0 +units=m +lon_0=' + str(float(lon_0)) + ' +lat_ts=90.0 +lat_0=90.0',
                      preserve_units=True)
        llcrnrlon = lon_0 - 45
        urcrnrlon = lon_0 + 135
        y_ = p(lon_0, boundinglat)[1]
        llcrnrlat = p(np.sqrt(2.) * y_, 0., inverse=True)[1]
        urcrnrlat = llcrnrlat
        llcrnrx, llcrnry = p(llcrnrlon, llcrnrlat)
        p = proj.Proj(
            '+proj=stere +R=6370997.0 +units=m +lon_0=' + str(float(lon_0)) + ' +lat_ts=90.0 +lat_0=90.0 +x_0=' + str(
                -llcrnrx) + ' +y_0=' + str(-llcrnry), preserve_units=True)
    return p


def readFB(grid_res, season, datapath, min_sie=0.15):
    """
    Read the input freeboard data (from all satellites) and sea ice extent (sie) mask.
    Returns:
        obs: array containing gridded freeboard information from all satellites
             of size (x,y,n,t), where n is the number of satellites (e.g., CS2 SAR,
             CS2 SARIN, S3A and S3B), and t is the number of days of observations
        sie_mask: array of daily sea ice extent of size (x,y,t), used to determine
                  which grid cells to interpolate to on a given day
        dates_trim: list dates (yyyymmdd) for which there are observations
    """


    f = open(datapath + '/CS2_SAR/CS2_SAR_dailyFB_' + str(grid_res) + 'km_' + season + '_season.pkl', 'rb')
    CS2_SAR = pickle.load(f)
    f = open(datapath + '/CS2_SARIN/CS2_SARIN_dailyFB_' + str(grid_res) + 'km_' + season + '_season.pkl', 'rb')
    CS2_SARIN = pickle.load(f)
    f = open(datapath + '/S3A/S3A_dailyFB_' + str(grid_res) + 'km_' + season + '_season.pkl', 'rb')
    S3A = pickle.load(f)
    f = open(datapath + '/S3B/S3B_dailyFB_' + str(grid_res) + 'km_' + season + '_season.pkl', 'rb')
    S3B = pickle.load(f)
    f = open(datapath + '/SIE_masking_' + str(grid_res) + 'km_' + season + '_season.pkl', 'rb')
    SIE = pickle.load(f)
    f.close()
    obs = []
    sie = []
    dates = []
    for key in CS2_SAR:
        if (key in CS2_SARIN) & (key in S3A) & (key in S3B):
            obs.append([CS2_SAR[key], CS2_SARIN[key], S3A[key], S3B[key]])
            sie.append(SIE[key])
            dates.append(key)
    obs = np.array(obs).transpose(2, 3, 1, 0)
    sie = np.array(sie).transpose(1, 2, 0)
    sie[sie < min_sie] = np.nan
    return obs, sie, dates


def GPR(x, y, xs, ell, sf2, sn2, mean, approx=False, M=None, returnprior=False):
    """
    Gaussian process regression function to predict radar freeboard
    Inputs:
            x: training data of size n x 3 (3 corresponds to x,y,time)
            y: training outputs of size n x 1 (observations of radar freeboard)
            xs: test inputs of size ns x 3
            ell: correlation length-scales of the covariance function (vector of length 3)
            sf2: scaling pre-factor for covariance function (scalar)
            sn2: noise variance (scalar)
            mean: prior mean (scalar)
            approx: Boolean, whether to use Nyström approximation method
            M: number of training points to use in Nyström approx (integer scalar)
    Returns:
            fs: predictive mean
            sfs2: predictive variance
            np.sqrt(Kxs[0][0]): prior variance
    """
    n = len(y)
    Kxsx = SGPkernel(x, xs=xs, ell=ell, sigma=sf2)
    Kxs = SGPkernel(xs, ell=ell, sigma=sf2)

    if approx:
        if M is None:
            M = int(n / 5)
        Ki, A = Nystroem(x, y, M=M, ell=ell, sf2=sf2, sn2=sn2)
        err = mdot([Kxsx.T, Ki, Kxsx])
    else:
        # this algo follows Algo 2.1 in Rasmussen (2006)
        Kx = SGPkernel(x, ell=ell, sigma=sf2) + np.eye(n) * sn2
        L = np.linalg.cholesky(Kx)
        A = np.linalg.solve(L.T, np.linalg.solve(L, y))
        v = np.linalg.solve(L, Kxsx)
        err = np.dot(v.T, v)

    fs = mean + np.dot(Kxsx.T, A)
    # taking the square root makes it standard deviation
    # TODO: update doc string
    sfs2 = np.sqrt((Kxs - err).diagonal())
    if returnprior:
        return fs, sfs2, np.sqrt(Kxs.diagonal()) # np.sqrt(Kxs[0][0])
    else:
        return fs, sfs2


def Nystroem(x, y, M, ell, sf2, sn2, seed=20, opt=False):
    """
    Nyström approximation for kernel machines, e.g., Williams
    and Seeger, 2001. Produce a rank 'M' approximation of K
    and find its inverse via Woodbury identity. This is a
    faster approach of making predictions, but performance will
    depend on the value of M.
    """
    np.random.seed(seed)
    n = len(y)
    randselect = sorted(np.random.choice(range(n), M, replace=False))
    Kmm = SGPkernel(x[randselect, :], ell=ell, sigma=sf2)
    Knm = SGPkernel(x, xs=x[randselect, :], ell=ell, sigma=sf2)
    Vi = np.eye(n) / sn2

    s, u = np.linalg.eigh(Kmm)
    s[s <= 0] = 1e-12
    s_tilde = n * s / M
    u_tilde = np.sqrt(M / n) * np.dot(Knm, u) / s
    L = np.linalg.cholesky(np.diag(1 / s_tilde) + mdot([u_tilde.T, Vi, u_tilde]))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.dot(u_tilde.T, Vi)))
    Ki = Vi - mdot([Vi, u_tilde, alpha])  # using Woodbury identity
    if opt:
        L_tilde = np.sqrt(s_tilde) * u_tilde
        det = np.linalg.slogdet(np.eye(M) * sn2 + np.dot(L_tilde.T, L_tilde))
        return Ki, np.atleast_2d(np.dot(Ki, y)).T, (det[0] * det[1]) / 2
    else:
        return Ki, np.atleast_2d(np.dot(Ki, y)).T


def SGPkernel(x, xs=None, grad=False, ell=1, sigma=1):
    """
    Return a Matern (3/2) covariance function for the given inputs.
    Inputs:
            x: training data of size n x 3 (3 corresponds to x,y,time)
            xs: test inputs of size ns x 3
            grad: Boolean whether to return the gradients of the covariance
                  function
            ell: correlation length-scales of the covariance function
            sigma: scaling pre-factor for covariance function
    Returns:
            sigma*k: scaled covariance function
            sigma*dk: scaled matrix of gradients
    """
    if xs is None:
        Q = squareform(pdist(np.sqrt(3.) * x / ell, 'euclidean'))
        k = (1 + Q) * np.exp(-Q)
        dk = np.zeros((len(ell), k.shape[0], k.shape[1]))
        for theta in range(len(ell)):
            q = squareform(pdist(np.sqrt(3.) * np.atleast_2d(x[:, theta] / ell[theta]).T, 'euclidean'))
            dk[theta, :, :] = q * q * np.exp(-Q)
    else:
        Q = cdist(np.sqrt(3.) * x / ell, np.sqrt(3.) * xs / ell, 'euclidean')
        k = (1 + Q) * np.exp(-Q)
    if grad:
        return sigma * k, sigma * dk
    else:
        return sigma * k


def SMLII(hypers, x, y, approx=False, M=None):
    """
    Objective function to minimise when optimising the model
    hyperparameters. This function is the negative log marginal likelihood.
    Inputs:
            hypers: initial guess of hyperparameters
            x: inputs (vector of size n x 3)
            y: outputs (freeboard values from all satellites, size n x 1)
            approx: Boolean, whether to use Nyström approximation method
            M: number of training points to use in Nyström approx (integer scalar)
    Returns:
            nlZ: negative log marginal likelihood
            dnLZ: gradients of the negative log marginal likelihood
    """
    ell = [np.exp(hypers[0]), np.exp(hypers[1]), np.exp(hypers[2])]
    sf2 = np.exp(hypers[3])
    sn2 = np.exp(hypers[4])
    n = len(y)
    Kx, dK = SGPkernel(x, grad=True, ell=ell, sigma=sf2)
    try:
        if approx:
            Ki, A, det = Nystroem(x, y, M=M, ell=ell, sf2=sf2, sn2=sn2, opt=True)
            nlZ = np.dot(y.T, A) / 2 + det + n * np.log(2 * np.pi) / 2
            Q = Ki - np.dot(A, A.T)
        else:
            L = np.linalg.cholesky(Kx + np.eye(n) * sn2)
            A = np.atleast_2d(np.linalg.solve(L.T, np.linalg.solve(L, y))).T
            nlZ = np.dot(y.T, A) / 2 + np.log(L.diagonal()).sum() + n * np.log(2 * np.pi) / 2
            Q = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n))) - np.dot(A, A.T)

        dnlZ = np.zeros(len(hypers))
        for theta in range(len(hypers)):
            if theta < 3:
                dnlZ[theta] = (Q * dK[theta, :, :]).sum() / 2
            elif theta == 3:
                dnlZ[theta] = (Q * (2 * Kx)).sum() / 2
            elif theta == 4:
                dnlZ[theta] = sn2 * np.trace(Q)
    except np.linalg.LinAlgError as e:
        nlZ = np.inf;
        dnlZ = np.ones(len(hypers)) * np.inf
    return nlZ, dnlZ


def SMLII_mod(hypers, x, y, approx=False, M=None, grad=True, use_log=True):
    """
    Objective function to minimise when optimising the model
    hyperparameters. This function is the negative log marginal likelihood.
    Inputs:
            hypers: initial guess of hyperparameters
            x: inputs (vector of size n x 3)
            y: outputs (freeboard values from all satellites, size n x 1)
            approx: Boolean, whether to use Nyström approximation method
            M: number of training points to use in Nyström approx (integer scalar)
    Returns:
            nlZ: negative log marginal likelihood
            dnLZ: gradients of the negative log marginal likelihood
    """
    # ell = [np.exp(hypers[0]), np.exp(hypers[1]), np.exp(hypers[2])]
    # sf2 = np.exp(hypers[3])
    # sn2 = np.exp(hypers[4])

    if use_log:
        ell = np.exp(hypers[:-2])
        sf2 = np.exp(hypers[-2])
        sn2 = np.exp(hypers[-1])
    else:
        ell = hypers[:-2]
        sf2 = hypers[-2]
        sn2 = hypers[-1]

    n = len(y)
    Kx, dK = SGPkernel(x, grad=True, ell=ell, sigma=sf2)
    try:
        if approx:
            Ki, A, det = Nystroem(x, y, M=M, ell=ell, sf2=sf2, sn2=sn2, opt=True)
            nlZ = np.dot(y.T, A) / 2 + det + n * np.log(2 * np.pi) / 2
            Q = Ki - np.dot(A, A.T)
        else:
            L = np.linalg.cholesky(Kx + np.eye(n) * sn2)
            A = np.atleast_2d(np.linalg.solve(L.T, np.linalg.solve(L, y))).T
            nlZ = np.dot(y.T, A) / 2 + np.log(L.diagonal()).sum() + n * np.log(2 * np.pi) / 2
            Q = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n))) - np.dot(A, A.T)

        if grad:
            dnlZ = np.zeros(len(hypers))
            for theta in range(len(hypers)):
                if theta < (len(hypers) - 2):
                    dnlZ[theta] = (Q * dK[theta, :, :]).sum() / 2
                elif theta == (len(hypers) - 2):
                    dnlZ[theta] = (Q * (2 * Kx)).sum() / 2
                elif theta == (len(hypers) - 1):
                    dnlZ[theta] = sn2 * np.trace(Q)
    except np.linalg.LinAlgError as e:
        nlZ = np.inf;
        dnlZ = np.ones(len(hypers)) * np.inf

    if grad:
        return nlZ, dnlZ
    else:
        return nlZ

def split_data2(sat, xFB, yFB, *args):
    # TODO: add a docstring
    # NOTE: the order of the values here is not the same a previous implementation
    # - would require a an axis swap?
    # identify the points that are NaN in sat data
    sat_nan = np.isnan(sat)
    # get the index along each dimension where not NaN
    non_nan = np.where(~sat_nan)
    # get the x and y location values for these
    x_train = xFB[non_nan[0], non_nan[1]]
    y_train = yFB[non_nan[0], non_nan[1]]
    # make a time index array
    t_train = np.arange(sat.shape[-1])[non_nan[3]]
    # freeboard observations
    z = sat[~sat_nan]

    # additional arguments are expected to be 2-D, same shape as xFB, yFB
    arg_out = [a[non_nan[0], non_nan[1]] for a in args]
    # prior = background[non_nan[0], non_nan[1]]

    return [x_train, y_train, t_train, z, *arg_out]


def split_data(sat, xFB, yFB, background):
    # wrapper for split data from different sat(ellite) data
    # TODO: tidy this up
    # TODO: add documentation

    # this next part loops over all T days of training data and appends all the inputs/outputs into long vectors
    x1 = []
    y1 = []
    t1 = []
    z1 = []
    x2 = []
    y2 = []
    t2 = []
    z2 = []
    x3 = []
    y3 = []
    t3 = []
    z3 = []
    x4 = []
    y4 = []
    t4 = []
    z4 = []
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    for dayz in range(sat.shape[3]):
        IDs_1 = np.where(~np.isnan(sat[:, :, 0, dayz]))
        IDs_2 = np.where(~np.isnan(sat[:, :, 1, dayz]))
        IDs_3 = np.where(~np.isnan(sat[:, :, 2, dayz]))
        IDs_4 = np.where(~np.isnan(sat[:, :, 3, dayz]))
        x1.extend(xFB[IDs_1])
        x2.extend(xFB[IDs_2])
        x3.extend(xFB[IDs_3])
        x4.extend(xFB[IDs_4])
        y1.extend(yFB[IDs_1])
        y2.extend(yFB[IDs_2])
        y3.extend(yFB[IDs_3])
        y4.extend(yFB[IDs_4])
        t1.extend(np.ones(np.shape(IDs_1)[1]) * dayz)
        t2.extend(np.ones(np.shape(IDs_2)[1]) * dayz)
        t3.extend(np.ones(np.shape(IDs_3)[1]) * dayz)
        t4.extend(np.ones(np.shape(IDs_4)[1]) * dayz)
        z1.extend(sat[:, :, 0, dayz][IDs_1])
        z2.extend(sat[:, :, 1, dayz][IDs_2])
        z3.extend(sat[:, :, 2, dayz][IDs_3])
        z4.extend(sat[:, :, 3, dayz][IDs_4])
        m1.extend(background[IDs_1])
        m2.extend(background[IDs_2])
        m3.extend(background[IDs_3])
        m4.extend(background[IDs_4])
    x_train = np.concatenate((x1, x2, x3, x4))
    y_train = np.concatenate((y1, y2, y3, y4))
    t_train = np.concatenate((t1, t2, t3, t4))
    z = np.concatenate((z1, z2, z3, z4))
    prior = np.concatenate((m1, m2, m3, m4))

    return x_train, y_train, t_train, z, prior


def load_data(datapath, grid_res, season,
              dates_to_datetime=False,
              trim_xy=None, **kwargs):
    # TODO: put x,y data loading in another function
    # grid
    # TODO: look into bin function to better understand how this works
    # shape (321,321)
    x = np.load(datapath + '/x_' + str(grid_res) + 'km.npy')  # zonal grid positions
    y = np.load(datapath + '/y_' + str(grid_res) + 'km.npy')  # meridional grid positions

    # x,y can be a different size (one longer in each dim) than obs, sie
    # - the below reduces the size
    if trim_xy:
        x = x[:-trim_xy, :-trim_xy]
        y = y[:-trim_xy, :-trim_xy]

    # object to convert x,y positions to longitude, latitude
    m = grid_proj(lon_0=360)
    lon, lat = m(x, y, inverse=True)

    # read in observations, sea ice and dates
    obs, sie, dates = readFB(grid_res, season, datapath, **kwargs)

    # another object to move lon, lat back to a (different?) x, y grid
    # - define a new projection to test interpolating 1 pixel
    mplot = grid_proj(llcrnrlon=-90, llcrnrlat=75, urcrnrlon=-152, urcrnrlat=82)
    # zonal & meridional positions for new projection
    xFB, yFB = mplot(lon, lat)

    if dates_to_datetime:
        dates = np.array([datetime.strptime(d, "%Y%m%d") for d in dates], dtype="datetime64[D]")
    else:
        dates = np.array(dates)

    return obs, sie, np.array(dates), xFB, yFB, lat, lon


def get_git_information():
    """
    helper function to get current git info
    - will get branch, current commit, last commit message
    - and the current modified file

    Returns
    -------
    dict with keys
        branch: branch name
        commit: current commit
        details: from last commit message
        modified: files modified since last commit, only provided if there were any modified files

    """
    # get current branch
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], shell=False)
        branch = branch.decode("utf-8").lstrip().rstrip()
    except Exception as e:
        branches = subprocess.check_output(["git", "branch"], shell=False)
        branches = branches.decode("utf-8").split("\n")
        branches = [b.lstrip().rstrip() for b in branches]
        branch = [re.sub("^\* ", "", b) for b in branches if re.search("^\*", b)][0]

    # current commit hash
    cur_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], shell=False)
    cur_commit = cur_commit.decode("utf-8").lstrip().rstrip()

    # last message
    last_msg = subprocess.check_output(["git", "log", "-1"], shell=False)
    last_msg = last_msg.decode("utf-8").lstrip().rstrip()
    last_msg = last_msg.split("\n")
    last_msg = [lm.lstrip().rstrip() for lm in last_msg]
    last_msg = [lm for lm in last_msg if len(lm) > 0]

    # modified files since last commit
    mod = subprocess.check_output(["git", "status", "-uno"], shell=False)
    mod = mod.decode("utf-8").split("\n")
    mod = [m.lstrip().rstrip() for m in mod]
    # keep only those that begin with mod
    mod = [re.sub("^modified:", "", m).lstrip() for m in mod if re.search("^modified", m)]

    out = {
        "branch": branch,
        "commit": cur_commit,
        "details": last_msg
    }

    # add modified files if there are any
    if len(mod) > 0:
        out["modified"] = mod

    return out


def WGS84toEASE2_New(lon, lat):
    """map from one mapping to another """
    # taken from /home/cjn/OI_PolarSnow/EASE/CS2S3_CPOM_bin_EASE.py
    EASE2 = "+proj=laea +lon_0=0 +lat_0=90 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    transformer = Transformer.from_crs(WGS84, EASE2)
    x, y = transformer.transform(lon, lat)
    return x, y


def EASE2toWGS84_New(x, y):
    EASE2 = "+proj=laea +lon_0=0 +lat_0=90 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    transformer = Transformer.from_crs(EASE2, WGS84)
    lon, lat = transformer.transform(x, y)
    return lon, lat


def move_to_archive(top_dir, file_names=None, suffix="", verbose=False):
    """
    Move file(s) matching a pattern to an 'archive' directory, adding suffixes if specified

    Parameters
    ----------
    top_dir: str, specifying path to existing directory containing files to archive
    file_names: str or list of str, default None. file names to move to archive
    suffix: str, default "", to be added to file names (before file type) before move to archive

    Returns
    -------
    None

    """

    assert os.path.exists(top_dir), f"top_dir:\n{top_dir}\ndoes not exist, expecting it to"

    assert file_names is not None, f"file_names not specified"

    # get the archive directory
    adir = os.path.join(top_dir, "archive")
    os.makedirs(adir, exist_ok=True)

    files_in_dir = os.listdir(top_dir)

    # check for files names
    for fn in file_names:
        if verbose:
            print("-"*10)
        # see if file names exists folder - has to be an exact match
        if fn in files_in_dir:

            _ = os.path.splitext(fn)
            # make a file name for the destination (add suffix before extension)
            fna = "".join([_[0], suffix, _[1]])

            # source and destination files
            src = os.path.join(top_dir, fn)
            dst = os.path.join(adir, fna)

            if verbose:
                print(f"{fn} moving to archive")
                print(f"file name in archive: {fna}")
            # move file
            shutil.move(src, dst)

        else:
            print(f"{fn} not found")


def plot_pcolormesh(ax, lon, lat, plot_data,
                    fig=None,
                    title=None,
                    vmin=None,
                    vmax=None,
                    cmap='YlGnBu_r',
                    cbar_label=None,
                    scatter=False,
                    **scatter_args):
    # TODO: finish with scatter option
    import cartopy.crs as ccrs
    import cartopy.feature as cfeat

    # ax = axs[j]
    ax.coastlines(resolution='50m', color='white')
    ax.add_feature(cfeat.LAKES, color='white', alpha=.5)
    ax.add_feature(cfeat.LAND, color=(0.8, 0.8, 0.8))
    ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())  # lon_min,lon_max,lat_min,lat_max

    if title:
        ax.set_title(title)

    if not scatter:
        s = ax.pcolormesh(lon, lat, plot_data,
                          cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(),
                          linewidth=0,
                          rasterized=True)
    else:
        non_nan = ~np.isnan(plot_data)
        s = ax.scatter(lon[non_nan],
                       lat[non_nan],
                       c=plot_data[non_nan],
                       cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(),
                       linewidth=0,
                       rasterized=True,
                       **scatter_args)

    if fig is not None:
        cbar = fig.colorbar(s, ax=ax, orientation='horizontal', pad=0.03, fraction=0.03)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=14)
        cbar.ax.tick_params(labelsize=14)


def date_str_to_datetime64(dates, format='%Y%m%d'):
    # this can be slow, replace with pd.to_datetime method?
    dates = [datetime.datetime.strptime(d, format)
             for d in dates]
    dates = np.array(dates, dtype='datetime64[D]')
    return dates



@nb.jit(nopython=True)
def rolling_mean(loc_obs, mean_array, window, trailing):
    # calculate the rolling mean
    # - if using trailing use up to (but excluding) the date
    if trailing:
        # if using a trailing window will only use prior days
        for i in range(window, len(mean_array)):
            # select date slice
            _ = loc_obs[:, (i - window): i, :]
            # if all are nan - i.e. missing, just forward fill (which could be nan)
            # TODO: here could apply symmetric window by just adjusting i back
            if np.isnan(_).all():
                mean_array[i] = mean_array[i-1]
            else:
                mean_array[i] = np.nanmean(_)
                # count_array[i] = (~np.isnan(_)).sum()
    # otherwise use symmetric window
    else:
        days_ahead = days_behind = window // 2
        # if window is even, make the days ahead one less
        if window % 2 == 0:
            days_ahead -= 1

        for i in range(days_behind + 1, len(mean_array) - days_ahead):
            # select date slice
            _ = loc_obs[:, (i - days_behind-1): (i+days_ahead), :]
            # if all are nan - i.e. missing, just forward fill (which could be nan)
            if np.isnan(_).all():
                mean_array[i] = mean_array[i-1]
            else:
                mean_array[i] = np.nanmean(_)
                # count_array[i] = (~np.isnan(_)).sum()
    return mean_array





if __name__ == "__main__":

    pass


