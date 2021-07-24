# Optimal Interpolation

This is the code used to generate daily pan-Arctic radar freeboard estimates from combined CryoSat-2 and Sentinel-3 satellite observations. See - [Gregory et al., 2021](https://tc.copernicus.org/articles/15/2857/2021/tc-15-2857-2021.pdf) for more information.

This code uses Gaussian Process Regression, with MPI implementation (mpi4py) in order to model 9 days of CryoSat-2 and Sentinel-3 gridded training data, and subsequently produce pan-Arctic estimates of radar freeboard on any given day.

# Citing 

Please cite the following DOI when using these data: [https://doi.org/10.5281/zenodo.5005979](https://doi.org/10.5281/zenodo.5005979)
