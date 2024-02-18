import xarray as xr
import xesmf as xe
import numpy as np
from global_land_mask import globe

EASEr = np.load('5km_EASE_2D.npz')
EASE = {'lon':EASEr['lon'],'lat':EASEr['lat']}
ocean = globe.is_ocean(EASE['lat'],EASE['lon'])
EASE['mask'] = ocean
dx,dy = EASE['lon'].shape

for year in range(2018,2020):
    print(year)
    obs_n = xr.open_dataset('NSIDC0051_SEAICE_PS_N25km_'+str(year)+'_v2.0.nc') #Data from https://doi.org/10.5067/MPYG15WAA4WX
    OBS = {'lon':obs_n['lon'].to_numpy(),'lat':obs_n['lat'].to_numpy()}
    obs_n['sic'] = obs_n.sic.where(obs_n.sic<=1).interpolate_na(dim='x',method='nearest',fill_value='extrapolate')

    regridder_n = xe.Regridder(OBS, EASE, method="bilinear")

    obs_regrid = regridder_n(obs_n['sic'].to_numpy())

    obs_regrid = xr.Dataset(data_vars=dict(sic=(['time','x','y'], obs_regrid)),coords=dict(time=obs_n['time'],x=np.arange(dx),y=np.arange(dy)))
    obs_regrid['lon'] = (('xT','yT'),EASE['lon'])
    obs_regrid['lat'] = (('xT','yT'),EASE['lat'])

    obs_regrid.to_netcdf('NSIDC0051_SEAICE_EASE5km_'+str(year)+'_v2.0.nc')
