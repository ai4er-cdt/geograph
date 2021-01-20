"""A collection of utility functions for data loading with rasterio"""

from typing import Optional
import numpy as np
import rasterio


def get_thumbnail(
    data: rasterio.io.DatasetReader,
    band_index: Optional[int] = 1,
    height: Optional[int] = 100,
    width: Optional[int] = None,
) -> np.ndarray:
    """
    Calculate a thumbnail for a given band of a rasterio dataset

    Args:
        data (rasterio.io.DatasetReader): rasterio dataset handle
        band_index (Optional[int], optional): The band index for which to caluclate the
            thumbnail. Defaults to 1.
        height (Optional[int], optional): The desired height of the thumbnail.
            If only the height is set, the width will be automatically determined
            from the datasets aspect ratio. Defaults to 100.
        width (Optional[int], optional): The desired width of the thumbnail.
            Defaults to None.

    Returns:
        np.ndarray: The 2D numpy array representing the thumbnail as calculated
            from nearest neighbour resampling.
    """

    aspect_ratio = data.height / data.width

    if height and not width:
        width = int(height / aspect_ratio)
    elif width and not height:
        height = int(width * aspect_ratio)

    assert height > 0 and width > 0, "Output height and/or width must be specified."

    return data.read(band_index, out_shape=(int(height), int(width)))
