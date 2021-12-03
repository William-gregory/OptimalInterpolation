### Generate pan-Arctic radar freeboard estimates on a given day
### Author: William Gregory
### Last updated: 03/12/2021

import numpy as np
import os
from scipy import stats
from scipy.spatial.distance import squareform,pdist,cdist
import scipy.optimize
import pickle
from mpi4py import MPI
import datetime
import itertools
from astropy.convolution import convolve, Gaussian2DKernel

COMM = MPI.COMM_WORLD

def split(container, count):
    """
    Function for dividing the number of tasks (container)
    across the number of available compute nodes (count)
    """
    return [container[_i::count] for _i in range(count)]

def readFB(grid_res,season):
    """
    Read the input freeboard data and sea ice extent (sie) mask.
    Returns:
        obs: array containing gridded freeboard information from all satellites
             of size (x,y,n,t), where n is the number of satellites (e.g., CS2 SAR,
             CS2 SARIN, S3A and S3B), and t is the number of days of observations
        sie_mask: array of daily sea ice extent of size (x,y,t), used to determine
                  which grid cells to interpolate to on a given day
        dates_trim: list dates (yyyymmdd) for which there are observations
    """
    f = open(datapath+'/CS2_SAR_dailyFB_'+str(grid_res)+'km_'+season+'_season.pkl','rb')
    CS2_SAR = pickle.load(f)
    f = open(datapath+'/CS2_SARIN_dailyFB_'+str(grid_res)+'km_'+season+'_season.pkl','rb')
    CS2_SARIN = pickle.load(f)
    f = open(datapath+'/S3A_dailyFB_'+str(grid_res)+'km_'+season+'_season.pkl','rb')
    S3A = pickle.load(f)
    f = open(datapath+'/S3B_dailyFB_'+str(grid_res)+'km_'+season+'_season.pkl','rb')
    S3B = pickle.load(f)
    f = open(datapath+'/SIE_masking_'+str(grid_res)+'km_'+season+'_season.pkl','rb')
    SIE = pickle.load(f)
    f.close()
    obs = []
    sie_mask = []
    dates = []
    for key in CS2_SAR:
        dates.append(key)
    dates = sorted(dates)
    dates_trim = []
    for date in dates:
        key = str(date)
        if (key in CS2_SARIN) & (key in S3A) & (key in S3B):
            obs.append([CS2_SAR[key],CS2_SARIN[key],S3A[key],S3B[key]])
            sie_mask.append(SIE[key])
            dates_trim.append(key)
    obs = np.array(obs).transpose(2,3,1,0)
    sie_mask = np.array(sie_mask).transpose(1,2,0)
    sie_mask[sie_mask<0.15] = np.nan
    return obs,sie_mask,dates_trim

def smooth(data,vmax,mask,std=1):
    """
    Function for applying a little bit of smoothing to the model 
    hyperparameters, to smooth out erroneous features.
    """
    data_smth = np.copy(data)
    data_smth[np.isinf(data_smth)] = np.nan
    data_smth[data_smth>vmax] = vmax
    data_smth = convolve(data_smth,Gaussian2DKernel(x_stddev=std,y_stddev=std))
    data_smth[data_smth==0] = np.nanmean(data_smth)
    data_smth[np.isnan(mask)] = np.nan
    return data_smth

def SGPkernel(x,xs=None,grad=False,ell=1,sigma=1):
    """
    Return a Matern (3/2) covariance function for the given inputs.
    Inputs:
            x: training data of size n x 3 (3 corresponds to x,y,time)
            xs: training inputs of size ns x 3
            grad: Boolean whether to return the gradients of the covariance
                  function
            ell: correlation length-scales of the covariance function
            sigma: scaling pre-factor for covariance function
    Returns:
            sigma*k: scaled covariance function
            sigma*dk: scaled matrix of gradients
    """
    if xs is None:
        Q = squareform(pdist(np.sqrt(3.)*x/ell,'euclidean'))
        k = (1 + Q) * np.exp(-Q)
        dk = np.zeros((len(ell),k.shape[0],k.shape[1]))
        for theta in range(len(ell)):
            q = squareform(pdist(np.sqrt(3.)*np.atleast_2d(x[:,theta]/ell[theta]).T,'euclidean'))
            dk[theta,:,:] = q * q * np.exp(-Q)
    else:
        Q = cdist(np.sqrt(3.)*x/ell,np.sqrt(3.)*xs/ell,'euclidean')
        k = (1 + Q) * np.exp(-Q)
    if grad:
        return sigma*k,sigma*dk
    else:
        return sigma*k

def SMLII(hypers,x,y,mX):
    """
    Objective function to minimise when optimising the model
    hyperparameters. This function is the negative log marginal likelihood.
    Inputs:
            hypers: initial guess of hyperparameters
            x: inputs (vector of size n x 3)
            y: outputs (freeboard values from all satellites, size n x 1)
            mX: prior mean value
    Returns:
            nlZ: negative log marginal likelihood
            dnLZ: gradients of the negative log marginal likelihood
    """
    ell = [np.exp(hypers[0]),np.exp(hypers[1]),np.exp(hypers[2])]
    sf2 = np.exp(hypers[3])
    sn2 = np.exp(hypers[4])
    n = len(y)
    Kx,dK = SGPkernel(x,grad=True,ell=ell,sigma=sf2)
    try:
        L = np.linalg.cholesky(Kx + np.eye(n)*sn2)
        A = np.atleast_2d(np.linalg.solve(L.T,np.linalg.solve(L,y-mX))).T
        nlZ = np.dot((y-mX).T,A)/2 + np.log(L.diagonal()).sum() + n*np.log(2*np.pi)/2

        Q = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n))) - np.dot(A,A.T)
        dnlZ = np.zeros(len(hypers))
        for theta in range(len(hypers)):
            if theta < 3:
                dnlZ[theta] = (Q*dK[theta,:,:]).sum()/2
            elif theta == 3:
                dnlZ[theta] = (Q*(2 * Kx)).sum()/2
            elif theta == 4:
                dnlZ[theta] = sn2*np.trace(Q)
    except np.linalg.LinAlgError as e:
        nlZ = np.inf ; dnlZ = np.ones(len(hypers))*np.inf
    return nlZ,dnlZ
        
def GPR3D(index,opt=True):
    """
    Gaussian Process Regression, the main interpolation function.
    Inputs:
            index: the index of the grid cell which will be interpolated
            opt: Boolean whether to optimise the hyperparameters
    Returns:
            fs: interpolated freeboard at location given by index
            sfs2: uncertainty in interpolated freeboard (1 standard deviation)
            lZ: log marginal likelihood
            lx: correlation length-scale hyperparameter in direction x
            ly: correlation length-scale hyperparameter in direction y
            lt: correlation length-scale hyperparameter in direction t
            sf2: variance pre-factor hyperparameter
            sn2: noise variance hyperparameter
    """
    idr = X_tree.query_ball_point(x=X[index,:], r=radius*1000)
    ID = (xy_train[None,:] == X[idr][:,None]).all(-1).any(0)
    inputs = np.array([x_train[ID],y_train[ID],t_train[ID]]).T
    outputs = z[ID]
    n = len(outputs)
    mX = np.ones(n)*mean
    Xs = np.atleast_2d(np.array([X[index,0],X[index,1],T_mid]))
    if opt:
        hypers = np.exp(scipy.optimize.minimize(SMLII,x0=x0,args=(inputs,outputs,mX),method='CG',jac=True).x)
        lx = hypers[0] ; ly = hypers[1] ; lt = hypers[2]
        sf2 = hypers[3] ; sn2 = hypers[4]
    else:
        IDxs = np.where((X[:,0]==X[index,0]) & (X[:,1]==X[index,1]))
        lx = ellXs[IDxs][0][0] ; ly = ellXs[IDxs][0][1] ; lt = ellXs[IDxs][0][2]
        sf2 = sf2xs[IDxs] ; sn2 = sn2xs[IDxs]
    Kx = SGPkernel(inputs,ell=[lx,ly,lt],sigma=sf2)
    Kxsx = SGPkernel(inputs,xs=Xs,ell=[lx,ly,lt],sigma=sf2)
    Kxs = SGPkernel(Xs,ell=[lx,ly,lt],sigma=sf2)
    try:
        L = np.linalg.cholesky(Kx + np.eye(n)*sn2)
        A = np.linalg.solve(L.T,np.linalg.solve(L,(outputs-mX)))
        lZ = - np.dot((outputs-mX).T,A)/2 - np.log(L.diagonal()).sum() - n*np.log(2*np.pi)/2
        v = np.linalg.solve(L,Kxsx)
        fs = mean + np.dot(Kxsx.T,A)
        sfs2 = np.sqrt((Kxs - np.dot(v.T,v)).diagonal())
        if opt:
            return fs[0],sfs2[0],lZ,lx,ly,lt,sf2,sn2
        else:
            return fs[0],sfs2[0]
    except np.linalg.LinAlgError as e:
        if opt:
            return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        else:
            return np.nan,np.nan

def save(dic,path):
    """
    Save dictionary containing interpolated freeboard, uncertainty
    and model hyperparameters (pan-Arctic) to disk
    """
    with open(path,'wb') as f:
        pickle.dump(dic,f,protocol=2)
    
grid_res = 25
datapath = os.path.expanduser('~') #this is the directory where this file currently exists
x = np.load(datapath+'/x_'+str(grid_res)+'km.npy')
y = np.load(datapath+'/y_'+str(grid_res)+'km.npy')

T=9 #number of days of data used to train the model
T_mid=T//2 #mid point location corresponding to the day which will be interpolated
radius = 300 #distance (in km) from each grid cell for which training data will be used for interpolation
obs,sie_mask,dates = readFB(grid_res,'2018-2019')
cs2_FYI = np.load(datapath+'/CS2_25km_FYI_20181101-20190428.npy')
day=1 #the day to interpolate will correspond to day+T_mid. Check which date this is with dates[day+T_mid]
mean=np.nanmean(cs2_FYI[:,:,day+16:day+25]).round(3) #mean of previous 9 days of CS2 FYI as prior ##YOU WILL NEED TO CHANGE THE RANGE (day+16:day+25) DEPENDING ON THE INPUT DATA
SIE = sie_mask[:,:,day+T_mid] #get the SIE for the day we wish to interpolate
sat = obs[:,:,:,day:day+T] #get the input observations +/- 4 days around the day we wish to interpolate
    
date = dates[day+T_mid] #the date which will be interpolated
x0 = [np.log(grid_res*1000),np.log(grid_res*1000),np.log(1.),np.log(1.),np.log(1.),np.log(.1)] #initial hyperparameter values

fs_grid = np.zeros(SIE.shape)*np.nan ; sfs2_grid = np.zeros(SIE.shape)*np.nan
lx_grid = np.zeros(SIE.shape)*np.nan ; ly_grid = np.zeros(SIE.shape)*np.nan ; lt_grid = np.zeros(SIE.shape)*np.nan
sf2_grid = np.zeros(SIE.shape)*np.nan ; sn2_grid = np.zeros(SIE.shape)*np.nan
lZ_grid = np.zeros(SIE.shape)*np.nan
x1 = [] ; y1 = [] ; t1 = [] ; z1 = []
x2 = [] ; y2 = [] ; t2 = [] ; z2 = []
x3 = [] ; y3 = [] ; t3 = [] ; z3 = []
x4 = [] ; y4 = [] ; t4 = [] ; z4 = []
for day in range(sat.shape[3]):
    IDs_1 = np.where(~np.isnan(sat[:,:,0,day]))
    IDs_2 = np.where(~np.isnan(sat[:,:,1,day]))
    IDs_3 = np.where(~np.isnan(sat[:,:,2,day]))
    IDs_4 = np.where(~np.isnan(sat[:,:,3,day]))
    x1.extend(x[IDs_1]) ; x2.extend(x[IDs_2]) ; x3.extend(x[IDs_3]) ; x4.extend(x[IDs_4])
    y1.extend(y[IDs_1]) ; y2.extend(y[IDs_2]) ; y3.extend(y[IDs_3]) ; y4.extend(y[IDs_4])
    t1.extend(np.ones(np.shape(IDs_1)[1])*day) ; t2.extend(np.ones(np.shape(IDs_2)[1])*day)
    t3.extend(np.ones(np.shape(IDs_3)[1])*day) ; t4.extend(np.ones(np.shape(IDs_4)[1])*day)
    z1.extend(sat[:,:,0,day][IDs_1]) ; z2.extend(sat[:,:,1,day][IDs_2])
    z3.extend(sat[:,:,2,day][IDs_3]) ; z4.extend(sat[:,:,3,day][IDs_4])
x_train = np.concatenate((x1,x2,x3,x4))
y_train = np.concatenate((y1,y2,y3,y4))
t_train = np.concatenate((t1,t2,t3,t4))
z = np.concatenate((z1,z2,z3,z4))
    
IDs = np.where(~np.isnan(SIE)) #grid cell locations which contain sea ice
X = np.array([x[IDs],y[IDs]]).T #put x,y positions of sea ice locations in a long vector
xy_train = np.array([x_train,y_train]).T
X_tree = scipy.spatial.cKDTree(X)
    
selected_variables = range(X.shape[0]) #the number of tasks to be done (i.e., the number of grid cells to be interpolated)

if COMM.rank == 0: #if master node
    splitted_jobs = split(selected_variables, COMM.size) #divide the number of grid cells across the number of available compute nodes
    print('start:',datetime.datetime.now())
else:
    splitted_jobs = None

scattered_jobs = COMM.scatter(splitted_jobs, root=0) #scatter the tasks to each of the computer nodes. E.g., if X.shape[0] = 5000 and we are using 10 nodes, then each node should receive 500 tasks

results = []
for index in scattered_jobs: #each computer node will execute this loop and send its segement of data to be interpolated to the GPR3D function, 1 grid cell at a time
    outputs = GPR3D(index)
    results.append(outputs) #store output interpolated values and hyperparameters in a list
results = COMM.gather(results, root=0) #gather all of the results together from all of the computer nodes.
        
if COMM.rank == 0: #tell the master node to compile the results into their own respective arrays and map back to the 2D domain
    res = {}
    fs = []
    sfs2 = []
    lZ = []
    lx = [] ; ly = [] ; lt = [] ; sf2 = [] ; sn2 = []
    results = list(itertools.izip_longest(*results))
    for r1 in results:
        for r2 in r1:
            if r2:
                fs.append(r2[0])
                sfs2.append(r2[1])
                lZ.append(r2[2])
                lx.append(r2[3])
                ly.append(r2[4])
                lt.append(r2[5])
                sf2.append(r2[6])
                sn2.append(r2[7])
    fs_grid[IDs] = np.array(fs)
    sfs2_grid[IDs] = np.array(sfs2)
    lZ_grid[IDs] = np.array(lZ)
    lx_grid[IDs] = np.array(lx)
    ly_grid[IDs] = np.array(ly)
    lt_grid[IDs] = np.array(lt)
    sf2_grid[IDs] = np.array(sf2)
    sn2_grid[IDs] = np.array(sn2)
    res[date+'_interp'] = fs_grid
    res[date+'_interp_error'] = sfs2_grid
    res[date+'_lZ'] = lZ_grid
    res[date+'_ell_x'] = lx_grid
    res[date+'_ell_y'] = ly_grid
    res[date+'_ell_t'] = lt_grid
    res[date+'_sf2'] = sf2_grid
    res[date+'_sn2'] = sn2_grid
    
    if grid_res == 25:
        std = 2
    else:
        std = 1
    res[date+'_ell_x_smth'] = smooth(res[date+'_ell_x'],2*radius*1000,SIE,std) #apply smoothing to each of the hyperparameters
    res[date+'_ell_y_smth'] = smooth(res[date+'_ell_y'],2*radius*1000,SIE,std)
    res[date+'_ell_t_smth'] = smooth(res[date+'_ell_t'],T,SIE,std)
    res[date+'_sf2_smth'] = smooth(res[date+'_sf2'],0.1,SIE,std)
    res[date+'_sn2_smth'] = smooth(res[date+'_sn2'],0.05,SIE,std)
else:
    res = None
    
res = COMM.bcast(res,root=0)
fs_smth = np.zeros(SIE.shape)*np.nan ; sfs2_smth = np.zeros(SIE.shape)*np.nan
ellXs = np.array([res[date+'_ell_x_smth'][IDs],res[date+'_ell_y_smth'][IDs],res[date+'_ell_t_smth'][IDs]]).T
sn2xs = res[date+'_sn2_smth'][IDs]
sf2xs = res[date+'_sf2_smth'][IDs]
results = []
for index in scattered jobs: #regenerate the predictions with the smoothed hyperparameters (now with optimisation=False)
    outputs = GPR3D(index,opt=False)
    results.append(outputs)
results = COMM.gather(results, root=0)

if COMM.rank == 0: #recompile results
    fs = []
    sfs2 = []
    results = list(itertools.izip_longest(*results))
    for r1 in results:
        for r2 in r1:
            if r2:
                fs.append(r2[0])
                sfs2.append(r2[1])
    fs_smth[IDs] = np.array(fs)
    sfs2_smth[IDs] = np.array(sfs2)
    res[date+'_interp_smth'] = fs_smth
    res[date+'_interp_error_smth'] = sfs2_smth
    print('finish:',datetime.datetime.now())
    save(res,datapath+'/CS2S3_'+date+'_'+str(grid_res)+'km.pkl') #save dictionary as pickle file
