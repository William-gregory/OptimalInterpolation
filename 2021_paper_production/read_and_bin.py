### Grid average along-track radar freeboard observations from CryoSat-2 and Sentinel-3
### Author: William Gregory
### Last updated: 07/01/2021

import numpy as np
import os
from mpl_toolkits.basemap import Basemap
from scipy import stats
import matplotlib as mpl
from scipy.interpolate import griddata
import pickle
import datetime
import glob

def save(dic,path):
    max_bytes = 2**31 -1
    bytes_out = pickle.dumps(dic,protocol=2)
    f = open(path,"wb")
    for idx in range(0, len(bytes_out), max_bytes):
        f.write(bytes_out[idx:idx+max_bytes])
    f.close()

def read_and_bin(SAT,months,days,version,grid_res):
    if SAT == 'CS2_SAR':
        datapath = '/cpnet/li1_cpdata/SATS/RA/CRY/L1B/GPOD_PROCESSED/SAR/processed/'
    elif SAT == 'CS2_SARIN':
        datapath = '/cpnet/li1_cpdata/SATS/RA/CRY/L1B/GPOD_PROCESSED/SARIN/processed/'
    elif SAT == 'S3A':
        datapath = '/cpnet/li2_cpdata/SATS/RA/S3A/L1B/GPOD_PROCESSED/processed/'
    elif SAT == 'S3B':
        datapath = '/cpnet/li2_cpdata/SATS/RA/S3B/L1B/GPOD_PROCESSED/processed/'
    bins = int(8e6/(grid_res*1000))
    FB = {}
    x = [] ; y = []
    k = 0
    for month in months:
        for day in range(days[k]):
            lon = [] ; lat = [] ; fb = []
            files = sorted(glob.glob(datapath+month+'/*'+month+str('%02d'%(day+1))+'*'+version+'.proc'))
            if len(files)>0:
                print(month+str('%02d'%(day+1)))
                for file in files:
                    data = np.genfromtxt(file)
                    valid = np.where((data[:,7]==2) & (data[:,4] >= -0.37) & (data[:,4] <= 0.63) & (~np.isnan(data[:,4])))
                    lon.extend(data[:,0][valid]) ; lat.extend(data[:,1][valid]) ; fb.extend(data[:,4][valid])
                fb = np.array(fb) ; lon = np.array(lon) ; lat = np.array(lat)
                x_vec,y_vec = m(lon,lat)
                binned_FB = stats.binned_statistic_2d(x_vec,y_vec,fb,\
                                       statistic=np.nanmean,bins=bins,range=[[0, 8e6], [0, 8e6]])
                xi,yi = np.meshgrid(binned_FB[1],binned_FB[2])
                x.append(xi) ; y.append(yi)
                FB[month+str('%02d'%(day+1))] = binned_FB[0].T
        k += 1
    save(FB,home+'/'+SAT+'_dailyFB_'+str(grid_res)+'km_'+season+'_season.pkl')
    if os.path.exists(home+'/x_'+str(grid_res)+'km.npy')==False:
        np.save(home+'/x_'+str(grid_res)+'km.npy',np.array(x).transpose(1,2,0)[:,:,0])
        np.save(home+'/y_'+str(grid_res)+'km.npy',np.array(y).transpose(1,2,0)[:,:,0])

m = Basemap(projection='npstere',boundinglat=60,lon_0=0, resolution='l',round=True)
grid_res = int(input('specify grid resolution in km:\n'))

home = os.path.expanduser('~')
SAT = str(input('specify which satellite to read and bin:\n'))
season = str(input('specify either 2018-2019 or 2019-2020:\n'))
if season == '2018-2019':
    months = ['201811','201812','201901','201902','201903','201904']
    days = [30,31,31,28,31,30]
    version = 'v1'
elif season == '2019-2020':
    months = ['201911','201912','202001','202002','202003','202004']
    days = [30,31,31,29,31,30]
    version = 'v3'
read_and_bin(SAT,months,days,version,grid_res)
