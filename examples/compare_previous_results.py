# compare previously generate results (using a version of PurePython engine)
# against the GPflow version


import os
import json
import re
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs

from OptimalInterpolation.sea_ice_freeboard import SeaIceFreeboard
from OptimalInterpolation.data_dict import DataDict
from OptimalInterpolation import get_data_path, read_key_from_config, get_images_path

from OptimalInterpolation.utils import plot_pcolormesh, grid_proj, WGS84toEASE2_New


# ---
# helper functions
# ---


def get_results_from_dir(res_dir, dates=None,
                         aux_dir=None,
                         cpom_dir=None,
                         big_grid_size=360):
    # TODO: remove commented sections of code below, and un used inputs
    # get the config file to determine how it was created
    with open(os.path.join(res_dir, "input_config.json"), "r") as f:
        config = json.load(f)

    # --
    # extract parameters
    # --

    # data_dir = config['data_dir']
    # data_dir = get_data_path() if data_dir == "package" else data_dir

    grid_res = config['grid_res']
    # season = config['season']
    #
    # if aux_dir is None:
    #     aux_dir = os.path.join(data_dir, "aux")
    #
    # if cpom_dir is None:
    #     cpom_dir = os.path.join(data_dir, "CS2S3_CPOM")

    # ---
    # SeaIceFreeboard
    # ---

    sifb = SeaIceFreeboard(grid_res=f"{grid_res}km",
                           length_scale_name=["x", "y", "t"])

    # --
    # load data - data is needed to build GPR
    # --

    # NOTE: it is crucial to load the same data
    # sifb.load_data(aux_data_dir=aux_dir,
    #                sat_data_dir=cpom_dir,
    #                grid_res=f"{grid_res}km",
    #                season=season)

    # --
    # read results - hyper parameters values, log likelihood
    # --

    res = sifb.read_results(res_dir,
                            file="results.csv",
                            grid_res_loc=grid_res,
                            unflatten=True,
                            grid_size=big_grid_size,
                            dates=dates)

    # --
    # read predictions - f*, f*_var, etc
    # --

    pre = sifb.read_results(res_dir,
                            file="prediction.csv",
                            grid_res_loc=grid_res,
                            unflatten=True,
                            grid_size=big_grid_size,
                            dates=dates)

    # ---
    # combine dicts
    # ---

    out = {**res, **pre}

    out['input_config'] = config

    # out['lon_grid'] = sifb.aux['lon']
    # out['lat_grid'] = sifb.aux['lat']

    return out


def parse_file_name(fn):
    _ = fn.split("_")
    out = {
        "date": _[1],
        "grid_res": _[2],
        "window": _[3],
        "radius": _[5]
    }
    return out


def stats_on_vals(vals, measure=None, name=None):
    """given a vals (np.array) get a DataFrame of some descriptive stats"""
    out = {}
    out['measure'] = measure
    out['size'] = vals.size
    out['num_not_nan'] = (~np.isnan(vals)).sum()
    out['num_inf'] = np.isinf(vals).sum()
    out['min'] = np.nanmin(vals)
    out['mean'] = np.nanmean(vals)
    out['max'] = np.nanmax(vals)
    out['std'] = np.nanstd(vals)

    qs = [0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95]
    quantiles = {f"q{q:.2f}": np.nanquantile(vals, q=q) for q in qs}
    out = {**out, **quantiles}

    columns = None if name is None else [name]
    return pd.DataFrame.from_dict(out, orient='index', columns=columns)


def compare_data_dict_table(d1, d2, date,
                            d1_measure=None,
                            d2_measure=None,
                            d1_col_name=None,
                            d2_col_name=None):
    d1 = d1.subset(select_dims={'date': date})
    d2 = d2.subset(select_dims={'date': date})

    d1_vals = np.squeeze(d1.vals)
    d2_vals = np.squeeze(d2.vals)

    _ = pd.concat([stats_on_vals(d1_vals, measure=d1_measure, name=d1_col_name),
                   stats_on_vals(d2_vals, measure=d2_measure, name=d2_col_name),
                   stats_on_vals(d1_vals - d2_vals, measure=f"{d1_col_name} - {d2_col_name}", name="diff")], axis=1)

    return _


def compare_data_dict_plot(d1, d2, date, lon, lat,
                           d1_measure=None,
                           d2_measure=None,
                           d1_col_name=None,
                           d2_col_name=None,
                           scatter=False,
                           cbar_label=None,
                           include_diff_cdf=False,
                           trim_cdf=False):
    if include_diff_cdf:

        fig = plt.figure(figsize=(10, 10))

        ax1 = fig.add_subplot(221, projection=ccrs.NorthPolarStereo())
        ax2 = fig.add_subplot(222, projection=ccrs.NorthPolarStereo())
        ax3 = fig.add_subplot(223, projection=ccrs.NorthPolarStereo())
        ax4 = fig.add_subplot(224)
        axes = [ax1, ax2, ax3, ax4]

        # fig, axes = plt.subplots(nrows=2, ncols=2,
        #                          figsize=(10, 10),
        #                          subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
        # axes = axes.flatten()
        # axes[3] = plt.subplot()
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3,
                                 figsize=(10, 5),
                                 subplot_kw=dict(projection=ccrs.NorthPolarStereo()))

        axes = axes.flatten()

    fig.suptitle(date)

    d1_ = d1.subset(select_dims={'date': date})
    d2_ = d2.subset(select_dims={'date': date})

    d1_vals = np.squeeze(d1_.vals)
    d2_vals = np.squeeze(d2_.vals)

    if isinstance(lon, DataDict):
        lon = lon.vals
    if isinstance(lat, DataDict):
        lat = lat.vals

    assert lat.shape == lon.shape
    assert d1_vals.shape == lon.shape
    assert d2_vals.shape == lon.shape

    # vmin = np.min([np.nanmin(d1_vals), np.nanmin(d2_vals)])
    # vmax = np.max([np.nanmax(d1_vals), np.nanmax(d2_vals)])

    vmin = np.min([np.nanquantile(d1_vals, q=0.005),np.nanquantile(d2_vals, q=0.005)])
    vmax = np.max([np.nanquantile(d1_vals, q=0.995),np.nanquantile(d2_vals, q=0.995)])

    plot_pcolormesh(ax=axes[0],
                    lon=lon,
                    lat=lat,
                    plot_data=d1_vals,
                    fig=fig,
                    title=f"{d1_measure}\n{d1_col_name}",
                    vmin=vmin,
                    vmax=vmax,
                    cmap='YlGnBu_r',
                    cbar_label=cbar_label,
                    scatter=scatter)

    plot_pcolormesh(ax=axes[1],
                    lon=lon,
                    lat=lat,
                    plot_data=d2_vals,
                    fig=fig,
                    title=f"{d2_measure}\n{d2_col_name}",
                    vmin=vmin,
                    vmax=vmax,
                    cmap='YlGnBu_r',
                    cbar_label=cbar_label,
                    scatter=scatter)

    # TODO: allow different dif options
    dif = d1_vals - d2_vals

    # dif_abs_max = np.nanmean(np.abs(dif))
    dif_abs_max = np.nanquantile(np.abs(dif), q=0.99)

    plot_pcolormesh(ax=axes[2],
                    lon=lon,
                    lat=lat,
                    plot_data=dif,
                    fig=fig,
                    title=f"diff\n{d1_col_name}-{d2_col_name}",
                    vmin=-dif_abs_max,
                    vmax=dif_abs_max,
                    cmap='bwr',
                    cbar_label=cbar_label,
                    scatter=scatter)

    if include_diff_cdf:

        # TODO: add quantile lines to cdf plot
        #  - 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95
        dif_sort = np.sort(dif[~np.isnan(dif)])
        qs = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])[::-1]
        qvals = np.quantile(dif_sort, q=qs)
        dif_min = np.min(dif_sort)

        cmap = plt.cm.get_cmap('viridis')

        cdf = np.arange(len(dif_sort)) / len(dif_sort)
        if trim_cdf:
            cdf_bool = (cdf >= np.quantile(cdf, q=0.001)) & (cdf <= np.quantile(cdf, q=0.999))
            plt_title = "CDF for dif"
        else:
            cdf_bool = (cdf >= cdf.min()) & (cdf <= cdf.max())
            plt_title = "CDF for dif - trimmed"


        axes[3].plot(dif_sort[cdf_bool], cdf[cdf_bool])
        for _, qv in enumerate(qvals):
            axes[3].hlines(y=qs[_], xmin=dif_min, xmax=qv,
                           linestyles='--', alpha=0.5, label=None, color=cmap(qs[_]))
            axes[3].vlines(x=qv, ymin=0, ymax=qs[_],
                           linestyles='--', alpha=0.5, label=f"{qs[_]:.2f}", color=cmap(qs[_]))

        axes[3].legend(loc=4)
        axes[3].set_title(plt_title)

    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    # ---
    # read in previously generated data (and interpolate to EASE grid)
    # ---

    sifb = SeaIceFreeboard()

    # ----
    # parameters
    # ----

    legacy_data = True
    grid_res = 50
    season = '2018-2019'
    radius = 300
    std = 50 / grid_res

    # legacy_data = True
    legacy_data_dir = "/home/buddy/workspace/Dissertation/LegacyData"

    # NOTE: it is crucial to load the same data - i.e. with the same grid
    sifb.load_aux_data(aux_data_dir=os.path.join(legacy_data_dir, "aux"),
                       season=season,
                       grid_res=f"{grid_res}km",
                       legacy_projection=legacy_data)

    # --
    # previous results (pkl_dir)
    # --

    dist_dir = "/mnt/hd1/data/dissertation/"
    pkl_dir = os.path.join(dist_dir, "interp")

    # read
    pdata = sifb.read_previous_data(pkl_dir=pkl_dir)

    # ----
    # GPflow results
    # ----

    # google drive location
    gdrive = read_key_from_config("directory_locations", "gdrive",
                                  example="gdrive")
    base_dir = os.path.join(gdrive, "Dissertation/refactored_legacy")
    sub_dir = f'radius300_daysahead4_daysbehind4_gridres{grid_res}_season2018-2019_coarsegrid1_holdout_boundlsFalse'
    results_dir = os.path.join(base_dir, sub_dir)

    # if using legacy data set big_grid_size=320 (instead of 360)
    resgp = get_results_from_dir(results_dir,
                                 big_grid_size=320)

    # add the mean to f* predictions
    resgp["f*"] = resgp["f*"] + resgp['mean']

    # ---
    # get auxiliary data - needed just for lon, lat
    # ---

    # season = resgp['input_config']['season']
    # grid_res = resgp['input_config']['grid_res']

    # ----
    # compare values
    # -----

    date = '20181201'
    d1_measure = 'interp'
    d2_measure = 'f*'
    d1 = pdata[d1_measure].copy()
    d2 = resgp[d2_measure].copy()

    # --
    # table summary
    # ---

    compdf = compare_data_dict_table(d1, d2, date,
                                     d1_measure=d1_measure,
                                     d2_measure=d2_measure,
                                     d1_col_name="Pure Python",
                                     d2_col_name="GPflow")

    lon = sifb.aux['lon']
    lat = sifb.aux['lat']

    # --
    # plot summary
    # ---

    compare_data_dict_plot(d1, d2, date, lon, lat,
                           d1_measure=d1_measure,
                           d2_measure=d2_measure,
                           d1_col_name="Pure Python",
                           d2_col_name="GPflow",
                           scatter=False,
                           cbar_label=None,
                           include_diff_cdf=True)

    plt.show()


    # ---
    # compare Marginal LogLikelihood
    # ---

    date = '20181201'
    d1_measure = 'lZ'
    d2_measure = 'marginal_loglikelihood'
    d1 = pdata[d1_measure].copy()
    d2 = resgp[d2_measure].copy()


    compdf = compare_data_dict_table(d1, d2, date,
                                     d1_measure=d1_measure,
                                     d2_measure=d2_measure,
                                     d1_col_name="Pure Python",
                                     d2_col_name="GPflow")


    compare_data_dict_plot(d1, d2, date, lon, lat,
                           d1_measure=d1_measure,
                           d2_measure=d2_measure,
                           d1_col_name="Pure Python",
                           d2_col_name="GPflow",
                           scatter=False,
                           cbar_label=None,
                           include_diff_cdf=True,
                           trim_cdf=True)

    plt.show()


