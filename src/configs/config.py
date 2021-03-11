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

    print("----------------- Options ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------   End -----------------")
    return cfg
