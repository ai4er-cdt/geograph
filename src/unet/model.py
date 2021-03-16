"""Module for the Pytorch Lightning Unet model."""
import multiprocessing as mp
from typing import List
import logging

import dask
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as losses
import torch
from src.constants import GWS_DATA_DIR
from src.unet.dataloader import LabelledSatelliteDataset
from src.unet.normalisers import ImagenetNormaliser, IdentityNormaliser
from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, SubsetRandomSampler

SENTINEL_DIR = GWS_DATA_DIR / "sentinel2_data"
SENTINEL_POLESIA_DIR = SENTINEL_DIR / "Polesia_10m"
SENTINEL_CHERNOBYL_DIR = SENTINEL_DIR / "Chernobyl_10m"


def get_unet_model(config):
    """Define and return the Unet model from the config."""
    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_depth=config.encoder_depth,
        encoder_weights=config.encoder_weights,
        decoder_channels=config.decoder_channels,
        decoder_attention_type=config.decoder_attention_type,
        in_channels=config.in_channels,
        classes=config.out_channels,
        activation=config.activation,
    )
    return model


class UNetModel(pl.LightningModule):
    """Unet model based on Pytorch Lightning."""

    def __init__(self, config) -> None:
        super().__init__()
        self.unet_logger = logging.getLogger("unet")
        self.save_hyperparameters(config)
        self.config = config
        self.model: torch.nn.Module = get_unet_model(config)
        self.loss = self.get_lossses([config.loss])
        self.eval_metric = self.get_lossses([config.eval_metric])
        self.normaliser = self.get_normaliser(config.normaliser)

    def configure_optimizers(self):
        self.unet_logger.info(
            "Configuring optimizer with learning rate %s", self.config.learning_rate
        )
        opt = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return opt

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        # .float() necessary to avoid type error with Double
        x, y = batch[0].float(), batch[1].float()
        y_hat = self(x)  # Calls self.forward(x)
        if torch.any(torch.isinf(y_hat)):
            return None
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch[0].float(), batch[1].float()
        y_hat = self(x)
        if torch.any(torch.isinf(y_hat)):
            return None
        return y, y_hat

    def validation_epoch_end(self, outputs: List) -> None:
        y_list, y_pred_list = [], []
        for y, y_pred in outputs:
            y_list.append(y)
            y_pred_list.append(y_pred)
        test_metric = self.eval_metric(torch.cat(y_pred_list), torch.cat(y_list))
        self.log(self.config.eval_metric, test_metric)

    def on_validation_epoch_start(self) -> None:
        self.model.train()

    def on_test_epoch_start(self):
        self.model.train()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List) -> None:
        y_list, y_pred_list = [], []
        for y, y_pred in outputs:
            y_list.append(y)
            y_pred_list.append(y_pred)
        test_metric = self.eval_metric(torch.cat(y_pred_list), torch.cat(y_list))
        self.log(f"Test: {self.config.eval_metric}", test_metric)

    def train_dataloader(self):
        """Load train dataset."""
        self.unet_logger.info("Train-loader: Loading training data")
        images_path = SENTINEL_POLESIA_DIR / "train"
        labels_path = (
            SENTINEL_POLESIA_DIR / "labels" / "polesia_labels_10m_train-tiled.tif"
        )

        if self.config.augment:
            aug_dict = {"rotation": True, "flip": True}
        else:
            aug_dict = {"rotation": False, "flip": False}
        if self.config.num_workers > 0:
            dask.config.set(scheduler="threads")
            train_set = LabelledSatelliteDataset(
                images_path=images_path,
                labels_path=labels_path,
                use_bands=self.config.use_bands[:],
                overlap_threshold=0.7,
                augmentations=aug_dict,
                normaliser=self.normaliser,
                chunks={"band": 1, "x": 256, "y": 256},
                n_classes=self.config.out_channels,
            )
            # pylint: disable=protected-access
            train_set._preload_chunk_handles()
            train_set.use_augmentations = self.config.augment
            dataloader = DataLoader(
                dataset=train_set,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
                pin_memory=True,  # asynchronous data transfer to the GPU
                multiprocessing_context=mp.get_context("fork"),
            )
            dask.config.set(scheduler="single-threaded")
        else:
            train_set = LabelledSatelliteDataset(
                images_path=images_path,
                labels_path=labels_path,
                use_bands=self.config.use_bands[:],
                overlap_threshold=0.7,
                augmentations=aug_dict,
                normaliser=self.normaliser,
                chunks={"band": 1, "x": 256, "y": 256},
                n_classes=self.config.out_channels,
            )
            train_set.use_augmentations = self.config.augment
            dataloader = DataLoader(
                dataset=train_set,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
                pin_memory=True,
            )  # asynchronous data transfer to the GPU
        return dataloader

    def val_dataloader(self):
        """Load validation dataset."""
        self.unet_logger.info("Valid-loader: Loading validation data")
        images_path = SENTINEL_POLESIA_DIR / "train"
        labels_path = (
            SENTINEL_POLESIA_DIR / "labels" / "polesia_labels_10m_valid-tiled.tif"
        )
        if self.config.num_workers > 0:
            dask.config.set(scheduler="threads")
            val_set = LabelledSatelliteDataset(
                images_path=images_path,
                labels_path=labels_path,
                use_bands=self.config.use_bands[:],
                overlap_threshold=0.9,
                normaliser=self.normaliser,
                chunks={"band": 1, "x": 256, "y": 256},
                n_classes=self.config.out_channels,
            )
            # pylint: disable=protected-access
            val_set._preload_chunk_handles()
            # Best practice to use shuffle=False for validation and testing.
            dataloader = DataLoader(
                dataset=val_set,
                batch_size=2,
                shuffle=False,
                sampler=SubsetRandomSampler(range(len(val_set) // 8)),
                num_workers=self.config.num_workers,
                pin_memory=True,
                multiprocessing_context=mp.get_context("fork"),
            )
            dask.config.set(scheduler="single-threaded")
        else:
            val_set = LabelledSatelliteDataset(
                images_path=images_path,
                labels_path=labels_path,
                use_bands=self.config.use_bands[:],
                overlap_threshold=0.9,
                normaliser=self.normaliser,
                chunks={"band": 1, "x": 256, "y": 256},
                n_classes=self.config.out_channels,
            )
            dataloader = DataLoader(
                dataset=val_set,
                batch_size=2,
                shuffle=False,
                sampler=SubsetRandomSampler(range(len(val_set) // 8)),
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        return dataloader

    def test_dataloader(self):
        """Load validation dataset."""
        self.unet_logger.info("Test-loader: Loading test data")
        images_path = SENTINEL_POLESIA_DIR / "train"
        labels_path = (
            SENTINEL_POLESIA_DIR / "labels" / "polesia_labels_10m_test-tiled.tif"
        )
        if self.config.num_workers > 0:
            dask.config.set(scheduler="threads")
            test_set = LabelledSatelliteDataset(
                images_path=images_path,
                labels_path=labels_path,
                use_bands=self.config.use_bands[:],
                overlap_threshold=0.9,
                normaliser=self.normaliser,
                chunks={"band": 1, "x": 256, "y": 256},
                n_classes=self.config.out_channels,
            )
            # pylint: disable=protected-access
            test_set._preload_chunk_handles()
            # Best practice to use shuffle=False for validation and testing.
            dataloader = DataLoader(
                dataset=test_set,
                batch_size=2,
                shuffle=False,
                sampler=SubsetRandomSampler(range(len(test_set) // 8)),
                num_workers=self.config.num_workers,
                pin_memory=True,
                multiprocessing_context=mp.get_context("fork"),
            )
            dask.config.set(scheduler="single-threaded")
        else:
            test_set = LabelledSatelliteDataset(
                images_path=images_path,
                labels_path=labels_path,
                use_bands=self.config.use_bands[:],
                overlap_threshold=0.9,
                normaliser=self.normaliser,
                chunks={"band": 1, "x": 256, "y": 256},
                n_classes=self.config.out_channels,
            )
            dataloader = DataLoader(
                dataset=test_set,
                batch_size=2,
                shuffle=False,
                sampler=SubsetRandomSampler(range(len(test_set) // 8)),
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        return dataloader

    def get_lossses(self, names: List[str]):
        """Return the loss function based on the config."""

        self.unet_logger.info("Selecting loss function: %s", names)
        if len(names) > 1:
            raise NotImplementedError
        else:
            name = names[0]
            if name == "Jaccard":
                return losses.JaccardLoss(
                    mode="multilabel",
                    log_loss=False,
                    from_logits=(self.config.activation is None),
                )
            elif name == "Dice":
                return losses.DiceLoss(
                    mode="multilabel",
                    log_loss=True,
                    from_logits=(self.config.activation is None),
                )
            elif name == "Focal":
                return losses.FocalLoss(mode="multilabel")
            elif name == "BCE":
                if self.config.activation is None:
                    return BCEWithLogitsLoss()
                else:
                    return BCELoss()

    def get_normaliser(self, name: str):
        """Return the normaliser based on the config."""

        self.unet_logger.info("Selecting normaliser: %s", name)
        if name == "imagenet":
            return ImagenetNormaliser()
        elif name is None:
            return IdentityNormaliser()
        else:
            raise NotImplementedError(f"Normaliser {name} is not implemented.")
