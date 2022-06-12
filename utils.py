# various helper functions

import numpy as np
from numpy.linalg import multi_dot as mdot
from scipy.spatial.distance import squareform, pdist, cdist
import pickle
import pyproj as proj

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


def readFB(grid_res, season, datapath):
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
    sie[sie < 0.15] = np.nan
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
        Kx = SGPkernel(x, ell=ell, sigma=sf2) + np.eye(n) * sn2
        L = np.linalg.cholesky(Kx)
        A = np.linalg.solve(L.T, np.linalg.solve(L, y))
        v = np.linalg.solve(L, Kxsx)
        err = np.dot(v.T, v)

    fs = mean + np.dot(Kxsx.T, A)
    sfs2 = np.sqrt((Kxs - err).diagonal())
    if returnprior:
        return fs, sfs2, np.sqrt(Kxs[0][0])
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

