"""This module provides a SatelliteImage class for interfacing with satellite images
that are stored in multiple shards on disk"""
from typing import Tuple, List
import os
import pathlib
import dask.array as da
import xarray as xr

# Type alias
XarrayDataArray = xr.core.dataarray.DataArray
DaskArray = da.core.Array


class SatelliteImage:
    """Class to combine a satellite image stored in several shards on disk."""

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        shard_paths: List[os.PathLike],
        chunks: dict = {"band": 1, "x": 256, "y": 256},
    ):
        """
        Class to combine a satellite image stored in several shards on disk.

        `SatelliteImage` loads handlers to the individual shards of a satellite
        image at the specified `shard_paths` and combines them into a Dask Array that
        can be lazily queried and loaded.

        IMPORTANT: Currently assumes that shards are named according to the format
            `path/to/file/custom_shard_name-XXXXXX-YYYYYYY.tif`
        where
            custom_shard_name is a valid path name without dashes `-`
            XXXXXX is a number such as e.g. 000003493 that gives the x-offset of the
                shards top left corner within the overall bounding box of the image
            YYYYYY is a number such as e.g. 000053025 that gives the y-offset of the
                shards top left corner within the overall bounding box of the image

        Args:
            shard_paths (List[os.PathLike]): Paths to the satellite image shards
            chunks (dict, optional): Chard specification to load for dask and xarray.
                Defaults to {"band": 1, "x": 256, "y": 256}.
        """
        self.shard_paths: List[os.PathLike] = [
            pathlib.Path(path) for path in sorted(shard_paths)
        ]
        self.shard_offsets: List[Tuple[float]] = [
            SatelliteImage._get_shard_offset_from_path(path)
            for path in self.shard_paths
        ]
        self.shard_handles: List[XarrayDataArray] = [
            xr.open_rasterio(path, chunks=chunks) for path in self.shard_paths
        ]
        self.crs: str = self.shard_handles[0].crs
        self.transform: Tuple[float] = self.shard_handles[0].transform
        self.band_names: Tuple[str] = self.shard_handles[0].descriptions

        # Combine shards for one composite image
        self._combine_shards()

    @staticmethod
    def _get_shard_offset_from_path(shard_path: os.PathLike) -> Tuple[int, int]:
        x, y = shard_path.stem.split("-")[1:]
        return int(x), int(y)

    def _combine_shards(self) -> None:
        """
        Combine all shards of an image into a dask array that can be evaluated lazily.
        """

        block_width = len([elem for elem in self.shard_offsets if elem[0] == 0])
        block_height = len([elem for elem in self.shard_offsets if elem[1] == 0])

        assert block_width * block_height == len(self.shard_paths)

        # Note: This is a slightly awkward creation of the block matrix like e.g.
        #   [[handle[0], handle[1], handle[2] ],
        #    [handle[3], handle[4], handle[5] ]]
        # because numpy reshaping of `shard_handlers` list caused problems with dask
        block_arrangement = [
            [self.shard_handles[row * block_width + col] for col in range(block_width)]
            for row in range(block_height)
        ]

        first_row = block_arrangement[0]
        first_col = [row[0] for row in block_arrangement]

        self._combined_x = xr.concat([handle.x for handle in first_row], "x")
        self._combined_y = xr.concat([handle.y for handle in first_col], "y")
        self._combined_image: DaskArray = da.block(block_arrangement)

    @property
    def x(self) -> XarrayDataArray:
        return self._combined_x

    @property
    def y(self) -> XarrayDataArray:
        return self._combined_y

    @property
    def combined_image(self) -> DaskArray:
        return self._combined_image

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.combined_image.shape

    @property
    def xy_to_index(self):
        raise NotImplementedError

    def __getitem__(self, multi_index) -> DaskArray:
        return self.combined_image[multi_index]
