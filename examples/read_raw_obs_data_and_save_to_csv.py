# read proc data - get along track satellite data into a single file
# - read in ".proc" files containing radar freeboard data
# - date-times of observations are inferred from file names
# - data is trimmed to only take values above and below min_obs and max_obs, respectively
# - data is saved in csv with format: raw_along_track_lon_lat_fb_{k}.csv, where {k} is a satellite name
# - based on previous work found in cpom_scripts/CS2S3_CPOM_bin_EASE.py
import datetime
import os
import re

import numpy as np
import pandas as pd

if __name__ == "__main__":

    pd.set_option("display.max_columns", 200)

    # TODO: change these as needed
    top_dir = "/mnt/hd1/data/freeboard/procdata"
    output_dir = "/mnt/hd1/data/freeboard/procdata"

    # min/max observations to take
    min_obs = -0.37
    max_obs = 0.63

    # top level directories containing satellite data
    # - within these dirs should be month stamped directories containing proc files
    # - it is assumed the files
    sat_dir = {
        "CS2_SAR": os.path.join(top_dir, "CS2_SAR"),
        "CS2_SARIN": os.path.join(top_dir, "CS2_SARIN"),
        "S3A": os.path.join(top_dir, "S3A"),
        "S3B": os.path.join(top_dir, "S3B")
    }

    # regular expression to identify the files to read
    # - could use v1\.proc$, or  v3\.proc$
    file_regex = "\.proc$"

    # store results in dictionary
    sat_res = {}

    # read in the data
    for sat, dir in sat_dir.items():
        print("*" * 1000)
        print(f"sat: {sat}")

        # increment over the month directories
        month_dirs = [os.path.join(dir, _) for _ in os.listdir(dir) if _.isdigit()]

        res = []
        for mdir in month_dirs:
            print("reading data in from")
            print(mdir)
            files = [_ for _ in os.listdir(mdir) if re.search(file_regex, _)]

            obs_count = 0
            for f in files:
                data = np.genfromtxt(os.path.join(mdir, f))

                # get the file datetime - interval
                dt0, dt1 = f.split("_")[-5], f.split("_")[-4]
                # convert to datetime(64)

                dt0 = np.datetime64(datetime.datetime.strptime(dt0, "%Y%m%dT%H%M%S"))
                dt1 = np.datetime64(datetime.datetime.strptime(dt1, "%Y%m%dT%H%M%S"))

                # split the difference in time by the number of observations
                #  - here assume the observations come in sequentially, equally spaced apart
                delta_t = (dt1 - dt0) / len(data)

                dt = dt0 + np.arange(len(data)) * delta_t

                # convert to seconds
                dt = dt.astype('datetime64[s]')

                # - previously converted to ms
                # express as day (since epoch) plus some fraction of day
                # TODO: should this be done elsewhere - to save space in data?
                # ms_in_day = 1000 * 60 * 60 * 24
                # day_frac = (dt - dt.astype('datetime64[D]')).astype('float') / ms_in_day
                # t = dt.astype('datetime64[D]').astype(float) + day_frac

                # HARDCODED: selection criteria for data - seemingly clipping
                # valid = np.where(
                #     (data[:, 7] == 2) & (data[:, 4] >= -0.37) & (data[:, 4] <= 0.63) & (~np.isnan(data[:, 4])))
                select = (data[:, 7] == 2) & (data[:, 4] >= min_obs) & (data[:, 4] <= max_obs) & (~np.isnan(data[:, 4]))
                data = data[select, :]
                _ = pd.DataFrame(data[:, [0, 1, 4]], columns=['lon', 'lat', 'fb'])

                # try to get the date from the file name
                # - prefer to have actual time of day: datetime
                _['datetime'] = dt[select]
                # _['t'] = t[select]

                obs_count += len(_)
                res.append(_)
            print(f"found: {obs_count} observations")

        df = pd.concat(res)
        # storing the satellite name as a row entry will use a fair bit of storage
        # df['sat'] = sat

        sat_res[sat] = df

    # write data to file

    for k, v in sat_res.items():
        out_file = os.path.join(output_dir, f"raw_along_track_lon_lat_fb_{k}.csv")
        print(f"writing data for sat: {k} to\n{out_file}")
        v.to_csv(out_file,
                 index=False)



