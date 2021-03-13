"""Data loader for general multispectral satellite imagery for our UNet model"""
import logging
import os
from typing import List, Dict, Tuple, Union, Optional
import random

import numpy as np
import torch
import torch.nn.functional
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import tqdm
from rasterio.plot import reshape_as_image

from src.unet.satellite_image import SatelliteImage
from src.unet.normalizers import NormalizerABC, ImagenetNormalizer

# pylint: disable=missing-function-docstring  #TODO: Docstring


class RotationChoice:
    """Rotate an image by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)


class SatelliteDataset(torch.utils.data.Dataset):
    """A class to store satellite datasets and generate tiles from them"""

    IMAGE_PATTERN = "*[0-9].tif"
    RGB_RESCALE_VALUE = 3000

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        images_path: os.PathLike,
        use_bands: List[int],
        tile_size: int = 256,
        normalizer: NormalizerABC = ImagenetNormalizer(),
        augmentations: Dict[str, bool] = {"rotation": False, "flip": False},
        chunks: dict = {"band": 10, "x": 256, "y": 256},
        logger: logging.Logger = logging.getLogger(),
        cache: Optional[bool] = None,
        lock: Optional[bool] = None,
    ):

        self.logger = logger
        self.tile_size = tile_size
        self.use_bands = use_bands
        self._tile_center_coords = None
        self._tile_label_overlaps = None
        self.normalizer = normalizer
        self.augmentations = augmentations
        self._load_satellite_images(images_path, chunks, cache, lock)
        self._construct_augmentor()
        self.is_preloaded = False

    def _load_satellite_images(
        self,
        images_path: os.PathLike,
        chunks: dict,
        cache: Optional[bool] = None,
        lock: Optional[bool] = None,
    ) -> None:

        # Scan all available shards
        self.logger.info("Sat-loading: Loading Satellite images")
        shard_paths = list(images_path.glob(SatelliteDataset.IMAGE_PATTERN))
        extract_image_name = lambda path: path.stem.split("-")[0]

        # Detect names of unique images
        self._names = np.unique(list(map(extract_image_name, shard_paths)))
        self.logger.info("Sat-loading: Detected %s unique images", len(self.names))

        # Load handlers for each unique image
        self.logger.info("Sat-loading: Loading image handles")
        gather_shards = lambda name: list(
            images_path.glob(name + SatelliteDataset.IMAGE_PATTERN)
        )
        self._images = [
            SatelliteImage(gather_shards(name), chunks=chunks, cache=cache, lock=lock)
            for name in self.names
        ]

    @property
    def names(self) -> List[str]:
        return self._names

    @property
    def images(self) -> List[SatelliteImage]:
        return self._images

    @property
    def tile_center_coords(self) -> np.ndarray:
        if self._tile_center_coords is None:
            self._calculate_center_tile_coords()
        return self._tile_center_coords

    @property
    def ntiles(self) -> int:
        return len(self.tile_center_coords) * len(self.images)

    def __len__(self) -> int:
        return self.ntiles

    def _preload_chunk_handles(self) -> None:
        if self.is_preloaded:
            return
        else:
            self.logger.info("Preloading file handles for multiprocessing.")
            for index in tqdm.tqdm(range(len(self)), desc="Preloading file handles."):
                image_index = self._get_image_index(index)
                row_min, col_min, _, _ = self._get_tile_bounds(index)

                (
                    self.images[image_index]
                    .combined_image[
                        self.use_bands, row_min : row_min + 1, col_min : col_min + 1
                    ]
                    .compute()
                )
            self.is_preloaded = True
            self.logger.info("File handles preloaded.")

    def _calculate_center_tile_coords(self) -> None:
        _, nrows, ncols = self.images[0].shape  # discarded value is nbands
        row_center_coords = np.arange(self.tile_size // 2, nrows, self.tile_size)
        col_center_coords = np.arange(self.tile_size // 2, ncols, self.tile_size)

        # create combination of all row_centers and column centers via meshgrid
        # Result:  [[arr1[0], arr2[0]],
        #           [arr1[0], arr2[1]],
        #                   ...
        #           [arr1[-1], arr2[-1]]]
        self._tile_center_coords = np.array(
            np.meshgrid(row_center_coords, col_center_coords)
        ).T.reshape(-1, 2)

    def _get_tile_bounds(self, index: int) -> Tuple[int, int, int, int]:

        tile_index = index % len(self.tile_center_coords)

        row_min, col_min = self.tile_center_coords[tile_index] - self.tile_size // 2
        row_max, col_max = self.tile_center_coords[tile_index] + self.tile_size // 2

        return row_min, col_min, row_max, col_max

    def _get_image_index(self, tile_index: int) -> int:
        return int(tile_index / len(self.tile_center_coords))

    def _get_tile_from_bounds(
        self,
        image_index: int,
        row_min: int,
        col_min: int,
        row_max: int,
        col_max: int,
        mode: str = "tensor",
    ) -> Union[np.ndarray, torch.Tensor]:

        # Extract raster from satellite image
        raster = (
            self.images[image_index]
            .combined_image[self.use_bands, row_min:row_max, col_min:col_max]
            .compute()
        )
        if mode == "raster":
            return raster

        # Normalize and convert to image
        rescaled_raster = np.clip(
            raster / SatelliteDataset.RGB_RESCALE_VALUE,
            a_min=0,
            a_max=1,
        )

        if mode == "image":
            return reshape_as_image(rescaled_raster)
        elif mode == "tensor":
            return torch.from_numpy(rescaled_raster)
        else:
            raise ValueError(
                f"Invalid mode {mode}. Must be one of `raster`, `image`, `tensor`"
            )

    def _construct_augmentor(self) -> None:
        # FIXME: Current augmentor only works with PIL images. Need to implement to
        # numpy arrays / torch to use

        augmentor_pipeline = []
        if self.augmentations["rotation"]:
            rotator = RotationChoice(angles=[0, 90, 180, 270])
            augmentor_pipeline.append(rotator)

        if self.augmentations["flip"]:
            augmentor_pipeline.append(transforms.RandomHorizontalFlip(0.5))
            augmentor_pipeline.append(transforms.RandomVerticalFlip(0.5))

        self.augmentor = transforms.Compose(augmentor_pipeline)

    def _augment(self, arr: np.ndarray, seed: int) -> np.ndarray:
        # FIXME: Current augmentor only works with PIL images. Need to implement to
        # numpy arrays / torch to use

        # Seed fixing such that image and label will match:
        # # c.f. https://github.com/pytorch/pytorch/issues/42331#issuecomment-667434293
        random.seed(seed)
        torch.manual_seed(seed)
        return self.augmentor(arr)

    def __getitem__(self, index: int) -> torch.Tensor:

        if index > self.ntiles - 1:
            raise KeyError(
                f"Index {index} out of bounds for axis of size {self.ntiles}"
            )

        image_index = self._get_image_index(index)
        tile_bounds = self._get_tile_bounds(index)

        return self._get_tile_from_bounds(image_index, *tile_bounds)


class LabelledSatelliteDataset(SatelliteDataset, torch.utils.data.Dataset):
    """A class to store labelled satellite datasets and generate tiles and labels"""

    def __init__(
        self,
        images_path: os.PathLike,
        labels_path: os.PathLike,
        use_bands: List[int],
        overlap_threshold: float = 0.7,
        use_one_hot: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(
            images_path=images_path,
            use_bands=use_bands,
            **kwargs,
        )
        self.use_one_hot = use_one_hot
        self.overlap_threshold = overlap_threshold
        self._load_labels(labels_path)

    def _load_labels(self, labels_path: os.PathLike) -> None:

        self.logger.info("Label-loading: Loading landcover labels")
        self.labels = SatelliteImage(labels_path)
        self.label_array = self.labels.combined_image.data.compute().squeeze()

        self.logger.info("Label-loading: Generating label mask")
        self.label_mask = self.label_array > 0

        self.logger.info("Label-loading: Generating one-hot encoding")
        self._encode_labels_as_one_hot()

    @property
    def labels_one_hot(self) -> torch.Tensor:
        return self._labels_one_hot

    @property
    def tile_center_coords(self) -> np.ndarray:
        if self._tile_center_coords is None:
            self._calculate_center_tile_coords()

            # Filter out valid tiles which have less than `overlap_threshold` labelled
            self._tile_center_coords = self._tile_center_coords[
                self.tile_label_overlaps > self.overlap_threshold
            ]
        return self._tile_center_coords

    @property
    def tile_label_overlaps(self) -> np.ndarray:
        if self._tile_label_overlaps is None:
            self._calculate_tile_label_overlaps()
        return self._tile_label_overlaps

    def _labelled_fraction(
        self, row_min: int, col_min: int, row_max: int, col_max: int
    ) -> float:

        masked_tile = self.label_mask[row_min:row_max, col_min:col_max]
        labelled_fraction = np.count_nonzero(masked_tile) / np.prod(masked_tile.shape)
        return labelled_fraction

    def _calculate_tile_label_overlaps(self) -> None:

        self._tile_label_overlaps = np.array(
            [
                self._labelled_fraction(
                    row_center - self.tile_size // 2,
                    col_center - self.tile_size // 2,
                    row_center + self.tile_size // 2,
                    col_center + self.tile_size // 2,
                )
                for row_center, col_center in self._tile_center_coords
            ]
        )

    def _encode_labels_as_one_hot(self) -> None:

        # Find number of unique labels for one hot enconding
        unique_labels = np.unique(self.label_array)
        self.logger.info("Label-encoding: Found %s unique labels", len(unique_labels))
        label_tensor = torch.from_numpy(self.label_array)

        # One hot encode axes: will be in H x W x C ordering
        labels_one_hot = torch.nn.functional.one_hot(
            label_tensor.long(), num_classes=len(unique_labels)
        )

        # Permute axes to have C x H x W ordering
        self._labels_one_hot = labels_one_hot.transpose(0, 2).transpose(1, 2)

    def _get_tile_label(self, row_min: int, col_min: int, row_max: int, col_max: int):
        if self.use_one_hot:
            return self.labels_one_hot[:, row_min:row_max, col_min:col_max]
        else:
            return self.label_array[row_min:row_max, col_min:col_max]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if index > self.ntiles - 1:
            raise KeyError(
                f"Index {index} out of bounds for axis of size {self.ntiles}"
            )

        image_index = self._get_image_index(index)
        tile_bounds = self._get_tile_bounds(index)

        return (
            self._get_tile_from_bounds(image_index, *tile_bounds),
            self._get_tile_label(*tile_bounds),
        )
