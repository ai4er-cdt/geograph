"""Main module to load and train the model. This should be the program entry point."""
import os
import pathlib

import hydra
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src import constants
from src.configs import config
from src.unet import model


def train_model(cfg):
    """Train model with PyTorch Lightning and log with Wandb."""
    # Set random seeds.
    seed_everything(cfg.seed)

    unet = model.UNetModel(cfg)

    # Setup logging and checkpointing.
    run_dir = pathlib.Path(constants.PROJECT_PATH / "logs" / cfg.run_name)
    # Force all runs to log to the herbie/gtc-bio project and allow anonymous
    # logging without a wandb account.
    wandb_logger = WandbLogger(
        name=cfg.run_name,
        save_dir=run_dir,
        entity="herbie",
        project="gtc-biodiversity",
        save_code=False,
        anonymous=True,
    )
    ckpt_dir: pathlib.Path = run_dir / "checkpoints"
    # Saves the top 2 checkpoints according to the test metric throughout
    # training.
    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}",
        period=cfg.checkpoint_freq,
        monitor="train_loss",
        save_top_k=2,
        mode="min",
    )

    # Instantiate Trainer
    trainer = Trainer(
        accelerator=cfg.parallel_engine,
        auto_select_gpus=cfg.cuda,
        gpus=cfg.gpus,
        benchmark=True,
        deterministic=True,
        checkpoint_callback=ckpt,
        prepare_data_per_node=False,
        max_epochs=cfg.epochs,
        logger=wandb_logger,
        log_every_n_steps=cfg.log_steps,
        val_check_interval=cfg.val_interval,
    )

    # Train model
    trainer.fit(unet)

    # Load best checkpoint
    unet = model.UNetModel.load_from_checkpoint(ckpt.best_model_path)
    trainer.test(unet)
    # Save weights from checkpoint
    statedict_path: pathlib.Path = run_dir / "saved_models" / "unet.pt"
    os.makedirs(os.path.dirname(statedict_path), exist_ok=True)
    torch.save(unet.model.state_dict(), statedict_path)


# Load hydra config from yaml files and command line arguments.
@hydra.main(
    config_path=os.path.join(constants.SRC_PATH, "configs"), config_name="config"
)
def main(cfg) -> None:
    """Load and validate the hydra config."""
    cfg = config.validate_config(cfg)
    train_model(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
