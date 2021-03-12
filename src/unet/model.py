"""Module for the Pytorch Lightning Unet model."""
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as losses
import torch
from src.constants import GWS_DATA_DIR
from src.unet.dataloader import LabelledSatelliteDataset
from torch.utils.data import DataLoader

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


def get_lossses(config):
    """Return the loss function based on the config."""
    if config.loss == "Jaccard":
        return losses.JaccardLoss(
            mode="multilabel", from_logits=(config.activation is not None)
        )
    if config.loss == "Focal":
        return losses.FocalLoss(mode="multilabel")


class UNetModel(pl.LightningModule):
    """Unet model based on Pytorch Lightning."""

    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model: torch.nn.Module = get_unet_model(config)
        self.loss = get_lossses(config)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return opt

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)  # Calls self.forward(x)
        # only support single loss at the moment
        loss = self.loss(y_hat, y)
        self.log("Loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        return y, y_hat

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
        # Use `pin_memory=True` here for asynchronous data transfer to the GPU,
        # speeding up data loading.
        dataloader = DataLoader(
            dataset=train_set,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        """Load validation dataset."""
        images_path = SENTINEL_POLESIA_DIR / "train"
        labels_path = GWS_DATA_DIR / "polesia_burned_superclasses_all_touched_10m.tif"
        rgb_bands = [2, 1, 0]
        train_set = LabelledSatelliteDataset(
            images_path=images_path, labels_path=labels_path, use_bands=rgb_bands
        )
        # Best practice to use shuffle=False for validation and testing.
        dataloader = DataLoader(
            dataset=train_set,
            batch_size=8,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
