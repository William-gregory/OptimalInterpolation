# combine results from cross validation into a single pdf


# example script for evaluating predictions
# - read in previously results (expect some data to be held out)


import os
import json

import numpy as np
import pandas as pd

from scipy.stats import shapiro, norm, skew, kurtosis


from OptimalInterpolation import get_data_path, get_path, read_key_from_config
from OptimalInterpolation.data_dict import DataDict
from OptimalInterpolation.sea_ice_freeboard import SeaIceFreeboard

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs

from OptimalInterpolation.utils import plot_pcolormesh

import seaborn as sns


def plot_cdf(ax, plt_data, legend_font_size=8):
    plt_sort = np.sort(plt_data[~np.isnan(plt_data)])
    # HARDCODED: quantiles to plot
    qs = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])[::-1]
    qvals = np.quantile(plt_sort, q=qs)
    dif_min = np.min(plt_sort)

    cmap = plt.cm.get_cmap('viridis')

    cdf = np.arange(len(plt_sort)) / len(plt_sort)
    # cdf_bool can be used for trimming
    # - here it does noting
    cdf_bool = (cdf >= cdf.min()) & (cdf <= cdf.max())
    ax.plot(plt_sort[cdf_bool], cdf[cdf_bool])
    for _, qv in enumerate(qvals):
        ax.hlines(y=qs[_], xmin=dif_min, xmax=qv,
                  linestyles='--', alpha=0.5, label=None, color=cmap(qs[_]))
        ax.vlines(x=qv, ymin=0, ymax=qs[_],
                  linestyles='--', alpha=0.5, label=f"{qs[_]:.2f}", color=cmap(qs[_]))

    ax.legend(loc=4, prop={"size": legend_font_size})


def plot_hist(ax, data, ylabel):
    sns.histplot(data=data, kde=True, ax=ax, rasterized=True)
    ax.set(ylabel=ylabel)
    ax.set(title="Histogram / Density")

    stats = {
        "mean": np.mean(data),
        "std": np.std(data),
        "skew": skew(data),
        "kurtosis": kurtosis(data),
        "num obs": len(data)
    }
    stats_str = "\n".join([f"{kk}: {vv:.2f}" for kk, vv in stats.items()])
    ax.text(0.2, 0.9, stats_str,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)


if __name__ == "__main__":

    pd.set_option("display.max_columns", 200)

    # directory to store results
    # output_base_dir = get_path("results", "local_results", "test")

    # ---
    # parameters
    # ---

    # parameters for result folder selection
    grid_res = 25
    season = "2018-2019"
    coarse_grid_spacing = 1
    hout = "S3A"

    # directory containing previous results
    gdrive_subdir = "Dissertation/refactored"
    gdrive = read_key_from_config("directory_locations", "gdrive",
                                  example="gdrive")

    # -----
    # initialise SeaIceFreeboard object
    # -----

    sifb = SeaIceFreeboard(grid_res=f"{grid_res}km",
                           length_scale_name=["x", "y", "t"])

    # load observation data, if it does not exist
    data_dir = get_data_path()
    sifb.load_data(aux_data_dir=os.path.join(data_dir, "aux"),
                   sat_data_dir=os.path.join(data_dir, "CS2S3_CPOM"),
                   grid_res=grid_res,
                   season=season)

    xval_dict = {}
    for hout in ["S3A", "S3B", "CS2_SAR"]:

        print("#" * 50)
        print(hout)

        # NOTE: holdouts denoted in directory name - multiple are separated by a |
        tmp_dir = f"radius300_daysahead4_daysbehind4_gridres{grid_res}_season{season}_coarsegrid{coarse_grid_spacing}_holdout{hout}_boundlsFalse"
        prev_results_dir = os.path.join(gdrive, gdrive_subdir, tmp_dir)

        if not os.path.exists(prev_results_dir):
            print(f"DIRECTORY:\n{prev_results_dir}\nDOES NOT EXIST\nSKIPPING HOLDOUT DATA:{hout}")
            continue

        assert os.path.exists(prev_results_dir)


        # ---
        # load the previous (prediction) results
        # ---

        prev_results = sifb.get_results_from_dir(
                            res_dir=prev_results_dir,
                            results_file="results.csv",
                            predictions_file="prediction.csv")

        # ----
        # get cross validation results on predictions
        # ----

        xval_dict[hout] = sifb.cross_validation_results(prev_results)

    # ----
    # summary - histograms, cdf and table
    # ----

    # TODO: all for use of regular differences

    use_norm_diff = True
    if use_norm_diff:
        measure = 'z'
    else:
        measure = 'diff'

    from OptimalInterpolation import get_images_path

    image_dir = get_images_path("cross_validation")
    os.makedirs(image_dir, exist_ok=True)

    # TODO: should sto

    # TODO: review file naming
    with PdfPages(get_images_path(image_dir, f"cross_val_results_example.pdf")) as pdf:

        # NOTE: if using regular differences only show two columns
        figsize = (12, 5 * len(xval_dict))
        fig, axes = plt.subplots(nrows=len(xval_dict), ncols=3,
                                 figsize=figsize,
                                 sharex=True)

        # add labels
        # https://stackoverflow.com/questions/51553545/matplotlib-tick-labels-disappeared-after-set-sharex-in-subplots
        for a in fig.axes:
            a.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,
                top=False,
                labelbottom=True)  # labels along the bottom edge are on

        # TODO: with these plots would like to have

        for idx, (k, v) in enumerate(xval_dict.items()):
            sat_name = k

            if len(axes.shape) > 1:
                axs = axes[idx, :]
            else:
                axs = axes

            # get all normalised differences
            z = v[measure]
            # get the dates data corresponds to
            dates_str = ", ".join(z.dims['date'].tolist())

            zvals = v[measure].vals
            non_nan = ~np.isnan(zvals)

            # ---
            # histogram
            # --

            plot_hist(ax=axs[0], data=zvals[non_nan],
                      ylabel=k)

            # ---
            # cdf
            # ---

            plot_cdf(ax=axs[1], plt_data=zvals[non_nan])
            axs[1].set(title="CDF")

            # ---
            # Q-Q plot?
            # ---

            # NOTE: if using regular differences exclude this

            # import scipy
            _ = zvals[non_nan]
            # HERE: shifted by 0.5 so don't try to get 0. or 1.0 quantile (occurs at -/+ inf)
            qs = (0.5+np.arange(len(_))) / len(_)
            norm_q = norm.ppf((0.5+np.arange(len(_))) / len(_))

            cmap = plt.cm.get_cmap('viridis')
            c = [cmap(_) for _ in qs]
            axs[2].scatter(norm_q, np.sort(_), c=c, alpha=0.5)
            axs[2].axline((norm_q.min(), norm_q.min()),
                          (norm_q.max(), norm_q.max()))
            axs[2].set(title="Q-Q plot")

            # ---
            # table ?
            # ---

            # pd.DataFrame.from_dict(stats, orient='index').to_string()


        fig.suptitle(f"Cross Validation on Grid: {grid_res}, Season: {season}"
                     f"\n {'Normalised ' if use_norm_diff else ''}Differences"
                     f"\ndates: {dates_str}",
                     y=0.99)

        plt.tight_layout()

        # save pdf here
        pdf.savefig()
        plt.show()

        # ----
        # Visualise per date
        # ----

        # ---
        # plot values on map
        # ---

        lon = sifb.aux['lon'].vals
        lat = sifb.aux['lat'].vals

        # for each holdout (sat data)
        for idx, (k, v) in enumerate(xval_dict.items()):

            dates = v['z'].dims['date']
            # NOTE: this will only get pairs - if odd one will get left out
            # TODO: add a check for adding odd number of dates
            date_pairs = [dates[(i*2):((i*2)+2)] for i in range(len(dates)//2)]

            for dpidx, dp in enumerate(date_pairs):

                print(f"date_pair: {dp}")

                figsize = (12, 12)

                # fig, axes = plt.subplots(nrows=2, ncols=2,
                #                          figsize=figsize,
                #                          subplot_kw=dict(projection=ccrs.NorthPolarStereo()))

                fig = plt.figure(figsize=figsize)
                ax1 = fig.add_subplot(221, projection=ccrs.NorthPolarStereo())
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223, projection=ccrs.NorthPolarStereo())
                ax4 = fig.add_subplot(224)

                fig.suptitle(f"Cross Val. Results for: {k}")

                axes = np.array([[ax1, ax2], [ax3, ax4]])

                for idx, date in enumerate(dp):

                    print(date)
                    axs = axes[idx, :]

                    # --
                    # plot the observations
                    # --

                    plt_data = v['z'].subset({'date': date})
                    plt_data = np.squeeze(plt_data.vals)
                    # TODO: change these is
                    vmin = -5
                    vmax = 5

                    plot_pcolormesh(ax=axs[0],  # axes[0],
                                    lon=lon,
                                    lat=lat,
                                    plot_data=plt_data,
                                    fig=fig,
                                    title=f"{k}\n{date}",
                                    vmin=vmin,
                                    vmax=vmax,
                                    cmap='bwr',
                                    cbar_label="norm diff",
                                    scatter=False)

                    # ---
                    # plot histogram / cdf / q-q plot
                    # ----

                    plot_hist(ax=axs[1],
                              data=plt_data[~np.isnan(plt_data)],
                              ylabel=k)

                # TODO: let the histograms share the same axis

                plt.tight_layout()
                pdf.savefig()
                plt.show()


