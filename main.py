import os
import hydra
from src import constants #, train_model
from src.configs import config


# Load hydra config from yaml files and command line arguments.
@hydra.main(config_path=os.path.join(constants.SRC_PATH, 'configs'), config_name="config")
def main(cfg) -> None:
    cfg = config.validate_config(cfg)
    # train_model(cfg)


if __name__ == "__main__":
    main()
