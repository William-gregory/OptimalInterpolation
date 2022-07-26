# compare two sets of results

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
from OptimalInterpolation import get_path, get_data_path, read_key_from_config, get_images_path

from OptimalInterpolation.utils import plot_pcolormesh, grid_proj, WGS84toEASE2_New


# ---
# helper functions
# ---

def clean_file_name(s):
    """remove unwanted characters from file name"""
    # https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename
    return "".join(x for x in s if (x.isalnum() or x in "._- "))


# def get_results_from_dir(res_dir, dates=None,
#                          results_file="results.csv",
#                          predictions_file="prediction.csv",
#                          file_suffix="",
#                          big_grid_size=360,
#                          results_data_cols=None,
#                          preds_data_cols=None
#                          ):
#     # TODO: remove commented sections of code below, and un used inputs
#     # get the config file to determine how it was created
#     with open(os.path.join(res_dir, f"input_config{file_suffix}.json"), "r") as f:
#         config = json.load(f)
#
#     # --
#     # extract parameters
#     # --
#
#     # data_dir = config['data_dir']
#     # data_dir = get_data_path() if data_dir == "package" else data_dir
#
#     grid_res = config['grid_res']
#     # season = config['season']
#     #
#     # if aux_dir is None:
#     #     aux_dir = os.path.join(data_dir, "aux")
#     #
#     # if cpom_dir is None:
#     #     cpom_dir = os.path.join(data_dir, "CS2S3_CPOM")
#
#     # ---
#     # SeaIceFreeboard
#     # ---
#
#     sifb = SeaIceFreeboard(grid_res=f"{grid_res}km",
#                            length_scale_name=["x", "y", "t"])
#
#     # --
#     # load data - data is needed to build GPR
#     # --
#
#     # NOTE: it is crucial to load the same data
#     # sifb.load_data(aux_data_dir=aux_dir,
#     #                sat_data_dir=cpom_dir,
#     #                grid_res=f"{grid_res}km",
#     #                season=season)
#
#     # --
#     # read results - hyper parameters values, log likelihood
#     # --
#
#     res = sifb.read_results(res_dir, file=results_file, grid_res_loc=grid_res, grid_size=big_grid_size, unflatten=True,
#                             dates=dates, file_suffix=file_suffix,
#                             data_cols=results_data_cols)
#
#     # --
#     # read predictions - f*, f*_var, etc
#     # --
#
#     pre = sifb.read_results(res_dir, file=predictions_file, grid_res_loc=grid_res, grid_size=big_grid_size,
#                             unflatten=True, dates=dates, file_suffix=file_suffix,
#                             data_cols=preds_data_cols)
#
#     # ---
#     # combine dicts
#     # ---
#
#     out = {**res, **pre}
#
#     out['input_config'] = config
#
#     # out['lon_grid'] = sifb.aux['lon']
#     # out['lat_grid'] = sifb.aux['lat']
#
#     return out


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
                           trim_cdf=False,
                           subtitle=None,
                           figsize=(10,10)):
    if include_diff_cdf:

        fig = plt.figure(figsize=figsize)

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
                                 figsize=figsize,
                                 subplot_kw=dict(projection=ccrs.NorthPolarStereo()))

        axes = axes.flatten()

    suptitle = date if subtitle is None else f"{date}\n{subtitle}"
    fig.suptitle(suptitle)

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
            plt_title = "CDF for diff - trimmed"


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

    grid_res = 50
    season = '2018-2019'
    radius = 300
    std = 50 / grid_res

    # TODO: add prefix for file name
    #  - or get config information to use for a name
    # image_subdir = "compare_GPflow_PurePythonCGJacTrue"
    image_subdir = "compare_regression"
    # suffix = "_clipsmooth"
    # suffix = "_replicate"

    suffix_list = ["", ""]
    coarse_grid_list = [1, 1]

    gdrive = read_key_from_config("directory_locations", "gdrive",
                                  example="gdrive")
    local_path = get_path()

    dir_list = [
        os.path.join(gdrive, "Dissertation/refactored"),
        os.path.join(gdrive, "Dissertation/test"),
        # os.path.join(local_path, "results/local_results/refactored_spaced"),
        # os.path.join(local_path, "results/local_results/test")
    ]

    d1_measure = 'f*'
    # d1_measure = 'interp_smth'
    d2_measure = 'f*'
    # d1_measure = 'lZ'
    # d2_measure = 'marginal_loglikelihood'

    results_data_cols = ['x', 'y', 'lon', 'lat',
                 'num_inputs',
                         # 'optimise_success',  # "run_time",
                 # "marginal_loglikelihood",
                 "ls_x", "ls_y", "ls_t",
                 "kernel_variance", "likelihood_variance",
                 ]

    preds_data_cols = ["f*", "f*_var", "y_var", "mean"]

    suffix_vs = '_vs_'.join(suffix_list)
    plot_file = clean_file_name(f"compare{suffix_vs}_{d1_measure}_{d2_measure}.pdf")


    # gdrive_subdir = "Dissertation/refactored_legacy"
    # gdrive_subdir = "Dissertation/refactored_spaced"
    os.makedirs(get_images_path(image_subdir), exist_ok=True)

    # legacy_data = True


    figsize = (14, 12)

    # legacy_data = True
    # big_grid_size = 320
    # data_dir = "/home/buddy/workspace/Dissertation/LegacyData"
    legacy_data = False
    big_grid_size = 360
    data_dir = get_data_path()

    # NOTE: it is crucial to load the same data - i.e. with the same grid
    sifb.load_aux_data(aux_data_dir=os.path.join(data_dir, "aux"),
                       season=season,
                       grid_res=f"{grid_res}km",
                       legacy_projection=legacy_data)

    # ----
    # GPflow results
    # ----


    reslist = []

    for i, base_dir in enumerate(dir_list):

        print("*" * 100)
        print(f"base_dir: {base_dir}")

        suffix = suffix_list[i]
        coarse_grid = coarse_grid_list[i]
        results_file = f"results{suffix}.csv"
        predictions_file = f"prediction{suffix}.csv"

        # google drive location
        # gdrive = read_key_from_config("directory_locations", "gdrive",
        #                               example="gdrive")
        # base_dir = os.path.join(gdrive, gdrive_subdir)
        sub_dir = f'radius300_daysahead4_daysbehind4_gridres{grid_res}_season2018-2019_coarsegrid{coarse_grid}_holdout_boundlsFalse'
        results_dir = os.path.join(base_dir, sub_dir)

        assert os.path.exists(results_dir), f"results_dir:\n{results_dir}\ndoes not exist!"

        # if using legacy data set big_grid_size=320 (instead of 360)
        resgp = sifb.get_results_from_dir(results_dir,
                                     results_file=results_file,
                                     predictions_file=predictions_file,
                                     big_grid_size=big_grid_size,
                                     file_suffix=suffix,
                                     results_data_cols=results_data_cols,
                                     preds_data_cols=preds_data_cols)

        # add the mean to f* predictions
        resgp["f*"] = resgp["f*"] + resgp['mean']

        reslist.append(resgp)

    # ---
    # get auxiliary data - needed just for lon, lat
    # ---

    # season = resgp['input_config']['season']
    # grid_res = resgp['input_config']['grid_res']

    # ----
    # compare values
    # -----

    # TODO: move this else where - be better with naming
    d1_col_name = f"{reslist[0]['input_config']['engine']} - Grid Spacing {coarse_grid_list[0]}"
    d2_col_name = f"{reslist[1]['input_config']['engine']} - Grid Spacing {coarse_grid_list[1]}"


    with PdfPages(get_images_path(image_subdir,plot_file)) as pdf:


        for date in reslist[0][d1_measure].dims['date']:

            print("*" * 75)
            print(f"date: {date}")

            d1 = reslist[0][d1_measure].copy()
            d2 = reslist[1][d2_measure].copy()

            # --
            # table summary
            # ---

            try:
                compdf = compare_data_dict_table(d1, d2, date,
                                                 d1_measure=d1_measure,
                                                 d2_measure=d2_measure,
                                                 d1_col_name=d1_col_name,
                                                 d2_col_name=d2_col_name)

            except AssertionError as e:
                print(e)
                print("skipping")
                continue

            lon = sifb.aux['lon']
            lat = sifb.aux['lat']

            print(compdf.style.to_latex())

            # --
            # plot summary
            # ---

            subtitle = f"diff - mean: {compdf.loc['mean', 'diff']: .2g}, std: {compdf.loc['std', 'diff']: .2g}"

            compare_data_dict_plot(d1, d2, date, lon, lat,
                                   d1_measure=d1_measure,
                                   d2_measure=d2_measure,
                                   d1_col_name=d1_col_name,
                                   d2_col_name=d2_col_name,
                                   subtitle=subtitle,
                                   scatter=False,
                                   cbar_label=None,
                                   include_diff_cdf=True,
                                   figsize=figsize)


            plot_file_png = clean_file_name(f"compare{suffix_vs}_{d1_measure}_{d2_measure}_{date}.png")
            print(plot_file_png)
            pdf.savefig()
            plt.savefig(get_images_path(image_subdir, plot_file_png))

            plt.show()

