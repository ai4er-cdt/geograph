"""
ani.py
============
"""
import copy
import numpy as np
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
from src.data_loading.landcover_plot_utils import classes_to_rgb
from src.utils import timeit
import imageio
import matplotlib.pyplot as plt


@timeit
def animate_prediction(x_da, y_da, pred_da, video_path="joint_val.mp4"):
    """
    This function animates the inputs, labels, and the corresponding predictions of the model.
    :param x_da: xarray.Dataarray, 3 bands, 4 seasons, 20 years
    :param y_da: xarray.Dataarray, 1 band, 20 years
    :param pred_da: xarray.Dataarray, 1 band, 20 years
    :param video_path: relative text path to output mp4 file.
    Based on code originally from Tom Anderson: tomand@bas.ac.uk.
    """

    def gen_frame_func(
        x_da, y_da, pred_da, mask=None, mask_type="contour", cmap="pink_r", figsize=15
    ):
        """
        Create imageio frame function for xarray.DataArray visualisation.

        Parameters:
        da (xr.DataArray): Dataset to create video of.

        mask (np.ndarray): Boolean mask with True over masked elements to overlay
        as a contour or filled contour. Defaults to None (no mask plotting).

        mask_type (str): 'contour' or 'contourf' dictating whether the mask is overlaid
        as a contour line or a filled contour.

        """
        cm = copy.copy(plt.get_cmap(cmap))
        cm.set_bad("gray")

        def make_frame(year):
            if len(x_da.band.values) == 3:
                fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
                    3, 2, figsize=(10, 10)
                )
            elif len(x_da.band.values) == 6:
                fig, (
                    (ax1, ax2),
                    (ax1b, ax2b),
                    (ax3, ax4),
                    (ax3b, ax4b),
                    (ax5, ax6),
                ) = plt.subplots(5, 2, figsize=(10, 17))
            else:
                assert False
            # da.sel(year=year).plot(cmap=cmap, clim=(min, max))

            x_da.isel(year=year, mn=0, band=slice(0, 3)).plot.imshow(ax=ax1)
            x_da.isel(year=year, mn=1, band=slice(0, 3)).plot.imshow(ax=ax2)
            x_da.isel(year=year, mn=2, band=slice(0, 3)).plot.imshow(ax=ax3)
            x_da.isel(year=year, mn=3, band=slice(0, 3)).plot.imshow(ax=ax4)

            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlabel("")

            if len(x_da.band.values) == 6:
                x_da.isel(year=year, mn=0, band=slice(3, 6)).plot.imshow(ax=ax1b)
                x_da.isel(year=year, mn=1, band=slice(3, 6)).plot.imshow(ax=ax2b)
                x_da.isel(year=year, mn=2, band=slice(3, 6)).plot.imshow(ax=ax3b)
                x_da.isel(year=year, mn=3, band=slice(3, 6)).plot.imshow(ax=ax4b)
                for ax in [ax1b, ax2b, ax3b, ax4b]:
                    ax.set_xlabel("")

            da = xr.DataArray(
                data=classes_to_rgb(y_da.isel(year=year).values),
                dims=["y", "x", "band"],
                coords=dict(
                    y=y_da.coords["y"].values,
                    x=y_da.coords["x"].values,
                    band=["red", "green", "blue"],
                    Y="esa_cci",
                ),
            )
            da.plot.imshow(ax=ax5)

            da = xr.DataArray(
                data=classes_to_rgb(
                    np.round_(pred_da.isel(year=year)).values.astype("int16")
                ),
                dims=["y", "x", "band"],
                coords=dict(
                    y=y_da.coords["y"].values,
                    x=y_da.coords["x"].values,
                    band=["red", "green", "blue"],
                    Y="predicted_classes",
                ),
            )
            da.plot.imshow(ax=ax6)
            if mask is not None:
                if mask_type == "contour":
                    ax.contour(mask, levels=[0.5, 1], colors="k")
                elif mask_type == "contourf":
                    ax.contourf(mask, levels=[0.5, 1], colors="k")
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

        return make_frame

    def xarray_to_video(
        x_da,
        y_da,
        pred_da,
        video_path,
        fps,
        mask_type="contour",
        video_dates=None,
        cmap="viridis",
        figsize=15,
    ):
        """
        Generate video of an xarray.DataArray. Optionally input a list of
        `video_dates` to show, otherwise the full set of time coordiantes
        of the dataset is used.
        """
        if video_dates is None:
            video_dates = [year for year in range(len(y_da.year.values))]
        make_frame = gen_frame_func(
            x_da=x_da,
            y_da=y_da,
            pred_da=pred_da,
            mask_type=mask_type,
            cmap=cmap,
            figsize=figsize,
        )
        imageio.mimsave(
            video_path,
            [make_frame(date) for date in tqdm(video_dates, desc=video_path)],
            fps=fps,
        )
        print("Video " + video_path + " made.")

    xarray_to_video(
        x_da,
        y_da,
        pred_da,
        video_path,
        mask_type=None,
        fps=5,
    )
