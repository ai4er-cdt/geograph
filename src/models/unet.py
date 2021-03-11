"""Module for the Pytorch Lightning Unet model."""
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as losses
import torch


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
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
