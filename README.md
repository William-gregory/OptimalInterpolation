# Optimal Interpolation

This is the code used to generate daily pan-Arctic radar freeboard estimates from combined CryoSat-2 and Sentinel-3 satellite observations. See - [https://doi.org/10.5194/tc-2020-371](https://tc.copernicus.org/preprints/tc-2020-371/tc-2020-371.pdf) for more information.

This code uses Gaussian Process Regression, with MPI implementation (mpi4py) in order to model 9 days of CryoSat-2 and Sentinel-3 gridded training data, and subsequently produce pan-Arctic estimates of radar freeboard on any given day.

![alt text](https://github.com/William-gregory/OptimalInterpolation/blob/main/images/Picture%201.png)
