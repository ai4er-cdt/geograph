"""This module defines normalizers for Satellite image data for UNet training"""
from abc import ABC, abstractmethod

import numpy as np
import torchvision.transforms as transforms
from torch import Tensor


class NormalizerABC(ABC):
    """Normalizer abstract base class.
    Serves as skeleton for the other Normalizer classes which are used in the project.
    Exposes the normalize abstract method which must be implemented by the subclasses.
    """

    def __init__(self):
        pass

    @abstractmethod
    def normalize(self, tensor: Tensor, **kwargs) -> Tensor:
        pass


class ImagenetNormalizer:
    """Imagenet normalizer that standardizes the input image using the ImageNet mean
    and std values.

    Values are taken from https://pytorch.org/docs/stable/torchvision/models.html

    Subtracts mean and divides by standard deviation
    """

    def __init__(self):
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])

        std_normalize = transforms.Normalize(
            mean=self.imagenet_mean, std=self.imagenet_std
        )
        self.transform = std_normalize

    # pylint: disable=unused-argument
    def normalize(self, tensor: Tensor, **kwargs) -> Tensor:
        """Return (tensor - imagenet_mean) / imagenet_std"""
        normalized_tensor = self.transform(tensor)
        return normalized_tensor


class IdentityNormalizer(NormalizerABC):
    """Identity normalizer that simply passes a tensor through without changing it."""

    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def normalize(self, tensor: Tensor, **kwargs) -> Tensor:
        """Return tensor unchanged."""
        return tensor
