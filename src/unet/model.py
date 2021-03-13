"""Module for the Pytorch Lightning Unet model."""
import multiprocessing as mp
from typing import List

import dask
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as losses
import torch
import tqdm
from src.constants import GWS_DATA_DIR
from src.unet.dataloader import LabelledSatelliteDataset
from torch.nn.modules.loss import BCEWithLogitsLoss
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
        self.save_hyperparameters(config)
        self.config = config
        self.model: torch.nn.Module = get_unet_model(config)
        self.loss = self.get_lossses([config.loss])
        self.eval_metric = self.get_lossses(["Jaccard"])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return opt

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch[0].float(), batch[1].float()
        # .float() necessary to avoid type error with Double
        y_hat = self(x)  # Calls self.forward(x)
        y = y.permute(0, 3, 1, 2).contiguous()
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch[0].float(), batch[1].float()
        y_hat = self(x)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y, y_hat

    def validation_epoch_end(self, outputs: List) -> None:
        y_list, y_pred_list = [], []
        for y, y_pred in outputs:
            y_list.append(y)
            y_pred_list.append(y_pred)
        test_metric = self.eval_metric(torch.cat(y_pred_list), torch.cat(y_list))
        self.log("Jaccard", test_metric)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def train_dataloader(self):
        """Load train dataset."""
        images_path = SENTINEL_POLESIA_DIR / "train"
        labels_path = GWS_DATA_DIR / "polesia_burned_superclasses_all_touched_10m.tif"
        rgb_bands = [2, 1, 0]
        train_set = LabelledSatelliteDataset(
            images_path=images_path, labels_path=labels_path, use_bands=rgb_bands
        )
        if self.config.num_workers > 0:
            dask.config.set(scheduler="threads")
            for i in tqdm.tqdm(range(len(train_set))):
                _ = train_set[i]
            dask.config.set(scheduler="single-threaded")
            dataloader = DataLoader(
                dataset=train_set,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True,  # asynchronous data transfer to the GPU
                multiprocessing_context=mp.get_context("fork"),
            )
        else:
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
        images_path = SENTINEL_POLESIA_DIR / "train"
        labels_path = GWS_DATA_DIR / "polesia_burned_superclasses_all_touched_10m.tif"
        rgb_bands = [2, 1, 0]
        val_set = LabelledSatelliteDataset(
            images_path=images_path, labels_path=labels_path, use_bands=rgb_bands
        )
        if self.config.num_workers > 0:
            dask.config.set(scheduler="threads")
            for i in tqdm.tqdm(range(len(val_set))):
                _ = val_set[i]
            dask.config.set(scheduler="single-threaded")
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
        else:
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
        return self.val_dataloader()

    def get_lossses(self, names: List[str]):
        """Return the loss function based on the config."""
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
                return BCEWithLogitsLoss()
