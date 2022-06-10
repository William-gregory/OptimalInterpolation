# Optimal Interpolation

This is a working repository for using Gaussian processes to combine and interpolate satellite altimetry data sets of radar freeboard. Here we explore the use of 'full' GPs to combine radar freeboards from CryoSat-2, Sentinel-3A and Sentinel-3B, in order produce daily radar freeboards. These results can be seen in the the following article [Gregory et al., 2021](https://tc.copernicus.org/articles/15/2857/2021/tc-15-2857-2021.pdf). The accompanying code for these results is contained within the 2021_paper_prodcution directory, and subseqeuently the corresponding quicklook data are available in the Quicklook Data directory.

# Citing 

Please cite the following DOI when using the quicklook data or 2021 production code: [https://doi.org/10.5281/zenodo.5005979](https://doi.org/10.5281/zenodo.5005979)

# Future work

We are currently working on a development version of the code (dev folder), which utilises the concept of sparse GPs (inducing points) using the python library GPflow. This is working towards an operational product which can be produced with minimal computational overheads.
