"""Module to validate the hydra config."""
import multiprocessing

import torch
from omegaconf import DictConfig, OmegaConf


def validate_config(cfg: DictConfig) -> DictConfig:
    """Validate the config and make any necessary alterations to the parameters."""
    if cfg.run_name is None:
        raise TypeError("The `run_name` argument is mandatory.")
    # Make sure num_workers isn't too high.
    core_count = multiprocessing.cpu_count()
    if cfg.num_workers > core_count * 2:
        cfg.num_workers = core_count
    if not cfg.cuda:
        cfg.gpus = 0
    if cfg.gpus <= 1:
        cfg.parallel_engine = None
    cfg.gpus = min(torch.cuda.device_count(), cfg.gpus)

    # Check that the number of decoder channels matches the depth of the encoder
    if len(cfg.decoder_channels) != cfg.encoder_depth:
        raise UserWarning("Encoder depth must match the number of decoder channels")

    # Check that number of bands to use for each image matches the net's input channels
    if len(cfg.use_bands) != cfg.in_channels:
        raise UserWarning(
            "Number of bands to use must agree with number of input channels."
        )

    # Check that if using image net we only work on RGB bands:
    if cfg.encoder_weights == "imagenet" and cfg.in_channels != 3:
        raise UserWarning("Imagenet initialisation only works with 3 in_channels.")

    print("----------------- Options ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------   End -----------------")
    return cfg
