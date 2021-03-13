"""This module contains code to downsample Raster images to lower resolution and to
generate thumbnails. Can be used to construct image pyramids of different resolutions
for enabling zooming in raster data on maps."""
import os
import multiprocessing as mp
import tqdm
from typing import List, Any, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.unet.labels import write_image


def downsample_image(
    image: rasterio.DatasetReader,
    bands: List[int],
    downsample_factor: int = 2,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, Any]:
    """
    Downsample the given bands of a raster image by the given downsample_fator.

    Args:
        image (rasterio.DatasetReader): Rasterio IO handle to the image
        bands (List[int]): The bands to downsample
        downsample_factor (int, optional): Factor by which the image will be
            downsampled. Defaults to 2.
        resampling (Resampling, optional): Resampling algorithm to use. Must be one of
            Rasterio's built-in resampling algorithms. Defaults to Resampling.bilinear.

    Returns:
        Tuple[np.ndarray, Any]: Return the resampled bands of the image as a numpy
            array together with the transform.
    """

    downsampled_image = image.read(
        bands,
        out_shape=(
            int(image.height / downsample_factor),
            int(image.width / downsample_factor),
        ),
        resampling=resampling,
    )

    transform = image.transform * image.transform.scale(
        (image.width / downsampled_image.shape[-1]),
        (image.height / downsampled_image.shape[-2]),
    )

    return downsampled_image, transform


def generate_downsample(
    file_path: os.PathLike,
    downsample_factor: int = 2,
    resampling: Resampling = Resampling.bilinear,
    overwrite: bool = False,
) -> None:
    """
    Generate downsample of the raster file at `file_path` and save it in same folder.

    Saved file will have appendix `_downsample_{downsample_factor}x.tif`

    Args:
        file_path (os.PathLike): The path to the raster image todownsample
        downsample_factor (int, optional): The downsampling factor to use.
            Defaults to 2.
        resampling (Resampling, optional): The resampling algorithm to use.
            Defaults to Resampling.bilinear.
        overwrite (bool, optional): Iff True, any existing downsampling file with the
            same downsampling factor will be overwritten. Defaults to False.
    """

    save_path = file_path.parent / (
        file_path.name.split(".")[0] + f"_downsample_{downsample_factor}x.tif"
    )
    if not overwrite and save_path.exists():
        return

    with rasterio.open(file_path) as image:
        downsampled_image, transform = downsample_image(
            image,
            image.indexes,
            downsample_factor=downsample_factor,
            resampling=resampling,
        )
        nbands, height, width = downsampled_image.shape

    write_image(
        downsampled_image,
        save_path,
        driver="GTiff",
        height=height,
        width=width,
        count=nbands,
        dtype="float64",
        crs=image.crs,
        transform=transform,
        nodata=image.nodata,
    )


def generate_thumbnail(
    file_path: os.PathLike,
    rgb_bands: Sequence[int] = (3, 2, 1),
    rescaling: int = 3000,
    downsample_factor: int = 16,
    resampling: Resampling = Resampling.bilinear,
    overwrite: bool = False,
) -> None:
    """
    Generate thumbnail of the raster file at `file_path` and save it in same folder.

    Saved file will have appendix `_thumbnail.tif`

    Args:
        file_path (os.PathLike): The path to the raster image to downsample
        rgb_bands (List[int], optional): The RGB bands in the order (red, green, blue)
            to produce a thumbnail that can be plotted as image.
        downsample_factor (int, optional): The downsampling factor to use.
            Defaults to 16.
        resampling (Resampling, optional): The resampling algorithm to use.
            Defaults to Resampling.bilinear.
        overwrite (bool, optional): Iff True, any existing thumbnail file will be
            overwritten. Defaults to False.
    """

    # print(f"Generating thumbnail for {file_path}\n")
    save_path = file_path.parent / (file_path.name.split(".")[0] + "_thumbnail.tif")
    if not overwrite and save_path.exists():
        # print("Thumbnail exists already.")
        return

    with rasterio.open(file_path) as image:
        downsampled_image, transform = downsample_image(
            image, rgb_bands, downsample_factor=downsample_factor, resampling=resampling
        )
        nbands, height, width = downsampled_image.shape
        crs = image.crs
        nodata = image.nodata

        downsampled_image = np.clip(downsampled_image / rescaling, a_min=0, a_max=1)

    # print(f"Saving thumbnail as {save_path}\n")
    write_image(
        downsampled_image,
        save_path,
        driver="GTiff",
        height=height,
        width=width,
        count=nbands,
        dtype="float64",
        crs=crs,
        transform=transform,
        nodata=nodata,
    )


def generate_thumbnail_parallel(files: List[os.PathLike], n_workers: int) -> None:
    """Generate thumbnail for all files in `files` via multiprocessing"""
    with mp.Pool(n_workers) as pool:
        with tqdm.tqdm(total=len(files)) as progress_bar:
            for _, _ in enumerate(pool.imap_unordered(generate_thumbnail, files)):
                progress_bar.update()


def _downsampler(argument_dict: dict) -> None:
    """Helper function for `generate_downsamples_parallel`"""
    file = argument_dict["file"]
    downsample_factor = argument_dict["downsample_factor"]
    return generate_downsample(file, downsample_factor)


def generate_downsamples_parallel(
    files: List[os.PathLike], downsample_factor: int, n_workers: int
) -> None:
    """Generate downsamples for all files in `files` via multiprocessing"""

    with mp.Pool(n_workers) as pool:
        with tqdm.tqdm(total=len(files)) as progress_bar:
            for _, _ in enumerate(
                pool.imap_unordered(
                    _downsampler,
                    [
                        {"file": file, "downsample_factor": downsample_factor}
                        for file in files
                    ],
                )
            ):
                progress_bar.update()
