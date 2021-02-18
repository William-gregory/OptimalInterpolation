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
    return [container[_i::count] for _i in range(count)]

def readFB(grid_res,season):
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
    data_smth = np.copy(data)
    data_smth[np.isinf(data_smth)] = np.nan
    data_smth[data_smth>vmax] = vmax
    data_smth = convolve(data_smth,Gaussian2DKernel(x_stddev=std,y_stddev=std))
    data_smth[data_smth==0] = np.nanmean(data_smth)
    data_smth[np.isnan(mask)] = np.nan
    return data_smth

def SGPkernel(x,xs=None,grad=False,ell=1,sigma=1):
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
    idr = X_tree.query_ball_point(x=X[index,:], r=radius*1000)
    ID = []
    for ix in range(len(X[idr])):
        ID.extend(np.where((x_train==X[idr][ix,0]) & (y_train==X[idr][ix,1]))[0])
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
        sfs2 = (Kxs - np.dot(v.T,v)).diagonal() + sn2
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
    with open(path,'wb') as f:
        pickle.dump(dic,f,protocol=2)
    
grid_res = 25
datapath = os.path.expanduser('~')
x = np.load(datapath+'/x_'+str(grid_res)+'km.npy')
y = np.load(datapath+'/y_'+str(grid_res)+'km.npy')

T=9
T_mid=T//2
radius = 300
res = {}
obs,sie_mask,dates = readFB(grid_res,'2018-2019')
cs2_FYI = np.load(datapath+'/CS2_25km_FYI_20181101-20190428.npy')
day=1 #December 1st 2018
mean=np.nanmean(cs2_FYI[:,:,day+16:day+25]).round(3) #mean of previous 9 days of CS2 FYI as prior
SIE = sie_mask[:,:,day+T_mid]
sat = obs[:,:,:,day:day+T]
    
date = dates[day+T_mid]
x0 = [np.log(grid_res*1000),np.log(grid_res*1000),np.log(1.),np.log(1.),np.log(1.),np.log(.1)]

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
    
IDs = np.where(~np.isnan(SIE))
X = np.array([x[IDs],y[IDs]]).T
X_tree = scipy.spatial.cKDTree(X)
    
selected_variables = range(X.shape[0])

if COMM.rank == 0:
    splitted_jobs = split(selected_variables, COMM.size)
    print(datetime.datetime.now())
    print('prior mean: ',str('%.3f'%mean))
else:
    splitted_jobs = None

scattered_jobs = COMM.scatter(splitted_jobs, root=0)

results = []
for index in scattered_jobs:
    outputs = GPR3D(index)
    results.append(outputs)
results = COMM.gather(results, root=0)
        
if COMM.rank == 0:
    print(datetime.datetime.now())
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
    res[date+'_ell_x_smth'] = smooth(res[date+'_ell_x'],600e3,SIE,std)
    res[date+'_ell_y_smth'] = smooth(res[date+'_ell_y'],600e3,SIE,std)
    res[date+'_ell_t_smth'] = smooth(res[date+'_ell_t'],9,SIE,std)
    res[date+'_sf2_smth'] = smooth(res[date+'_sf2'],0.1,SIE,std)
    res[date+'_sn2_smth'] = smooth(res[date+'_sn2'],0.05,SIE,std)

fs_smth = np.zeros(SIE.shape)*np.nan ; sfs2_smth = np.zeros(SIE.shape)*np.nan
ellXs = np.array([res[date+'_ell_x_smth'][IDs],res[date+'_ell_y_smth'][IDs],res[date+'_ell_t_smth'][IDs]]).T
sn2xs = res[date+'_sn2_smth'][IDs]
sf2xs = res[date+'_sf2_smth'][IDs]
results = []
for index in scattered jobs:
    outputs = GPR3D(index,opt=False)
    results.append(outputs)
results = COMM.gather(results, root=0)

if COMM.rank == 0:
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
    
    save(res,datapath+'/FBinterp_'+date+'_'+str(grid_res)+'km_'+str(T)+'dayT_'+str(radius)+'kmiter_nonzeroMatern.pkl')