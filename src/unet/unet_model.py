"""Module for the Pytorch Lightning Unet model."""
import logging
import multiprocessing as mp
from typing import List, Optional, Union

import dask
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as losses
import torch
import torch.nn.functional as F
from pytorch_lightning import metrics
from segmentation_models_pytorch.losses._functional import soft_jaccard_score
from segmentation_models_pytorch.losses.constants import (
    BINARY_MODE,
    MULTICLASS_MODE,
    MULTILABEL_MODE,
)
from src.constants import GWS_DATA_DIR
from src.unet.dataloader import LabelledSatelliteDataset
from src.unet.normalisers import IdentityNormaliser, ImagenetNormaliser
from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
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


class UNetModel(pl.LightningModule):
    """Unet model based on Pytorch Lightning."""

    def __init__(self, config) -> None:
        super().__init__()
        self.unet_logger = logging.getLogger("unet")
        self.save_hyperparameters(config)
        self.config = config
        self.model: torch.nn.Module = get_unet_model(config)
        self.loss = self.get_lossses([config.loss])
        self.eval_metrics: List = self.get_lossses(list(config.eval_metrics))
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
        loss = self.loss[0](y_hat, y)
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
        self.get_eval_metrics(
            mode="Validation",
            preds=torch.cat(y_pred_list),
            targets=torch.cat(y_list).int(),
        )

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
        self.get_eval_metrics(
            mode="Test", preds=torch.cat(y_pred_list), targets=torch.cat(y_list).int()
        )

    def get_eval_metrics(self, mode: str, preds, targets):
        for i, metric_fn in enumerate(self.eval_metrics):
            result = metric_fn(preds, targets)
            self.log(f"{mode}: {self.config.eval_metrics[i]}", result)

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
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        return dataloader

    def get_lossses(self, names: Union[str, List[str]]):
        """Return the loss function based on the config."""
        self.unet_logger.info("Selecting loss function: %s", names)

        if isinstance(names, str):
            names = [names]

        losses_list = []
        for name in names:
            if name == "Jaccard":
                losses_list.append(
                    losses.JaccardLoss(
                        mode="multilabel",
                        log_loss=False,
                        from_logits=(self.config.activation is None),
                    )
                )
            elif name == "Dice":
                losses_list.append(
                    losses.DiceLoss(
                        mode="multilabel",
                        log_loss=False,
                        from_logits=(self.config.activation is None),
                    )
                )
            elif name == "Focal":
                losses_list.append(losses.FocalLoss(mode="multilabel"))
            elif name == "BCE":
                if self.config.activation is None:
                    losses_list.append(BCEWithLogitsLoss())
                else:
                    losses_list.append(BCELoss())
            elif name == "Accuracy":
                losses_list.append(metrics.Accuracy().to(torch.device("cuda")))
            elif name == "IoU":
                losses_list.append(
                    CustomJaccardLoss(
                        mode="multilabel",
                        log_loss=False,
                        from_logits=(self.config.activation is None),
                        reduction="none",
                    )
                )
        return losses_list

    def get_normaliser(self, name: str):
        """Return the normaliser based on the config."""
        self.unet_logger.info("Selecting normaliser: %s", name)
        if name == "imagenet":
            return ImagenetNormaliser()
        elif name is None:
            return IdentityNormaliser()
        else:
            raise NotImplementedError(f"Normaliser {name} is not implemented.")


class CustomJaccardLoss(losses.JaccardLoss):
    """Custom Jaccard Loss that allows for elementwise IoU output."""

    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
        reduction: str = "elementwise_mean",
    ):
        """Jaccard loss for image segmentation task.

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation.
            By default, all channels are included.
            log_loss: If True, loss computed as `- log(jaccard_coeff)`,
            otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            ignore_index: Label that indicates ignored pixels (does not
            contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division
            error (denominator will be always greater or equal to eps)
            reduction: Whether or not to take the mean of the result.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(mode, classes, log_loss, from_logits, smooth, eps)

        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate Jaccard allowing for outputs for each class."""

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and
            # does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = soft_jaccard_score(
            y_pred,
            y_true.type(y_pred.dtype),
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        if self.classes is not None:
            loss = loss[self.classes]
        if self.reduction == "elementwise_mean":
            return loss.mean()
        else:
            return loss
