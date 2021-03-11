"""Functions to prepare the labels from landcover vector data for training with the
UNet model"""
import os
import numpy as np
import rasterio
from rasterio import features
from geopandas import GeoDataFrame


def write_image(arr: np.array, save_path: os.PathLike, **raster_meta) -> None:
    """
    Write a Geotiff to disk with the given raster metadata.

    Convenience function that automatically sets permissions correctly.

    Args:
        arr (np.array): The data to write to disk in geotif format
        save_path (os.PathLike): The path to write to
    """

    with rasterio.open(save_path, "w", **raster_meta) as target:
        target.write(arr)
    save_path.chmod(0o664)  # Allow group workspace users access


def zeros_tif_like(
    raster_data: rasterio.io.Dataset,
    save_path: os.PathLike,
    nbands: int = 1,
    dtype: str = "uint8",
) -> None:
    """
    Create Geotiff with `0` in all bands with the shape and meta data of `raster_data`

    Args:
        raster_data (rasterio.io.Dataset): Rasterio IO handle
        save_path (os.PathLike): Path to save the zeros-tif at
        nbands (int, optional): Number of bands to create and fill with 0.
            Defaults to 1.
        dtype (str, optional): Datatype to use. Defaults to "uint8".
    """

    height, width = raster_data.shape[1:]
    empty_template = np.zeros((nbands, height, width), dtype=dtype)

    write_image(
        empty_template,
        save_path,
        driver="GTiff",
        height=height,
        width=width,
        count=nbands,
        dtype=dtype,
        crs=raster_data.crs,
        transform=raster_data.transform,
    )


def burn_vector_to_raster(
    vector_data: GeoDataFrame,
    colname: str,
    template_path: os.PathLike,
    save_path: os.PathLike,
    **kwargs
) -> None:
    """TODO: write docstring"""

    with rasterio.open(template_path) as raster_data:
        out_arr = raster_data.read(1)
        transform = raster_data.transform
        meta = raster_data.meta.copy()
        meta.update(compress="lzw")

    with rasterio.open(save_path, "w+", **meta) as out:
        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = (
            (geom, value)
            for geom, value in zip(vector_data["geometry"], vector_data[colname])
        )

        burned = features.rasterize(
            shapes=shapes, out=out_arr, transform=transform, **kwargs
        )
        out.write_band(1, burned)
