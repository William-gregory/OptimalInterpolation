
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import seaborn as sns
from scipy.stats import shapiro, norm, skew, kurtosis

from OptimalInterpolation.data_dict import DataDict
from OptimalInterpolation.utils import plot_pcolormesh



def plot_hist(ax, data,
              title="Histogram / Density",
              ylabel=None,
              xlabel=None,
              select_bool=None,
              stats_values=None,
              stats_loc = (0.2, 0.9)):

    hist_data = data if select_bool is None else data[select_bool]
    sns.histplot(data=hist_data, kde=True, ax=ax, rasterized=True)
    ax.set(ylabel=ylabel)
    ax.set(xlabel=xlabel)
    ax.set(title=title)

    # provide stats if stats values is not None
    if stats_values is not None:
        stats = {
            "mean": np.mean(data),
            "std": np.std(data),
            "skew": skew(data),
            "kurtosis": kurtosis(data),
            "num obs": len(data)
        }

        stats_values = [stats_values] if isinstance(stats_values, str) else stats_values
        for sv in stats_values:
            assert sv in stats, f"stats_values: {sv} not in stats: {list(stats.keys)}"
        stats = {_: stats[_] for _ in stats_values}
        stats_str = "\n".join([f"{kk}: {vv:.2f}" if isinstance(vv, float) else f"{kk}: {vv:d}"
                               for kk, vv in stats.items()])
        ax.text(stats_loc[0], stats_loc[1], stats_str,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)



def compare_data_dict_plot(d1, d2, date, lon, lat,
                           dif_type="abs",
                           norm_std=None,
                           norm_for_diff=None,
                           d1_measure=None,
                           d2_measure=None,
                           d1_col_name=None,
                           d2_col_name=None,
                           scatter=False,
                           cbar_label=None,
                           include_diff_cdf=False,
                           include_diff_hist=False,
                           trim_cdf=False,
                           subtitle=None,
                           trim_to_quantile=0.005,
                           diff_trim_quantile=0.01,
                           vmin=None,
                           vmax=None,
                           dif_vmin=None,
                           dif_vmax=None,
                           figsize=(10,10)):

    if include_diff_cdf | include_diff_hist:

        fig = plt.figure(figsize=figsize) # constrained_layout=True

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

    if vmin is None:
        vmin = np.min([np.nanquantile(d1_vals, q=trim_to_quantile),
                       np.nanquantile(d2_vals, q=trim_to_quantile)])
    if vmax is None:
        vmax = np.max([np.nanquantile(d1_vals, q=1-trim_to_quantile),
                       np.nanquantile(d2_vals, q=1-trim_to_quantile)])

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
    # dif = d1_vals - d2_vals
    # TODO: allow different dif options
    if dif_type == "abs":
        dif = d1_vals - d2_vals
    elif dif_type == "rel":
        dif = d1_vals / d2_vals - 1
    elif dif_type == "abs_norm":

        assert norm_for_diff is not None, f"dif_type: {dif_type}, but norm_for_diff is None"
        # norm_for_diff should be a DataDict
        nfd = norm_for_diff.subset(select_dims={'date': date})
        nfd_vals = np.squeeze(nfd.vals)
        dif = (d1_vals - d2_vals) / nfd_vals


    # if norm_std provided - use to 'normalise' the difference
    if norm_std is not None:
        if isinstance(norm_std, DataDict):
            norm_std = norm_std.subset(select_dims={'date': date}).vals
        norm_std = np.squeeze(norm_std)
        assert dif.shape == norm_std.shape, \
            f"dif array shape: {dif.shape} is not the equal to norm_std: {norm_std.shape}"

        dif /= norm_std

    # dif_abs_max = np.nanmean(np.abs(dif))
    # TODO: set the diff quantile
    dif_abs_max = np.nanquantile(np.abs(dif), q=(1-diff_trim_quantile))

    if dif_vmin is None:
        dif_vmin = - dif_abs_max
    if dif_vmax is None:
        dif_vmax = dif_abs_max

    plt_title_norm_diff = "" if norm_std is None else "\ndiffs normalised"

    plot_pcolormesh(ax=axes[2],
                    lon=lon,
                    lat=lat,
                    plot_data=dif,
                    fig=fig,
                    title=f"{dif_type} diff\n{d1_col_name}-{d2_col_name}" + plt_title_norm_diff,
                    vmin=dif_vmin,
                    vmax=dif_vmax,
                    cmap='bwr',
                    cbar_label=cbar_label,
                    scatter=scatter)

    if include_diff_cdf | include_diff_hist:
        # TODO: make into a sub function
        # TODO: add quantile lines to cdf plot
        #  - 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95
        dif_sort = np.sort(dif[~np.isnan(dif)])
        qs = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])[::-1]
        qvals = np.quantile(dif_sort, q=qs)
        dif_min = np.min(dif_sort)

        cmap = plt.cm.get_cmap('viridis')

        cdf = np.arange(len(dif_sort)) / len(dif_sort)
        if trim_cdf:
            trim_size = 0.001
            cdf_bool = (cdf >= np.quantile(cdf, q=0.001)) & (cdf <= np.quantile(cdf, q=1-trim_size))
            plt_title = f"{'CDF' if include_diff_cdf else 'Histogram'} for diff - trimmed top and bottom {trim_size * 100 :.2f}%"
        else:
            cdf_bool = (cdf >= cdf.min()) & (cdf <= cdf.max())
            plt_title = f"{'CDF' if include_diff_cdf else 'Histogram'} for diff"


        if include_diff_cdf:
            axes[3].plot(dif_sort[cdf_bool], cdf[cdf_bool])
            for _, qv in enumerate(qvals):
                axes[3].hlines(y=qs[_], xmin=dif_min, xmax=qv,
                               linestyles='--', alpha=0.5, label=None, color=cmap(qs[_]))
                axes[3].vlines(x=qv, ymin=0, ymax=qs[_],
                               linestyles='--', alpha=0.5, label=f"{qs[_]:.2f}", color=cmap(qs[_]))

            axes[3].legend(loc=4)
            axes[3].set_title(plt_title)
        # otherwise plot a histogram
        else:
            dif_sort_std = np.std(dif_sort)
            dfm = np.mean(dif_sort)
            select_bool = ((dif_sort - dfm) >= - 3 * dif_sort_std) & \
                          ((dif_sort - dfm) <= 3 * dif_sort_std)
            plot_hist(axes[3], dif_sort[select_bool], ylabel="")


    plt.tight_layout()
    # plt.show()

