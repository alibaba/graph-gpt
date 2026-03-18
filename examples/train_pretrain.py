import sys

sys.path.insert(0, ".")

from src.training import TrainingPipeline, launch
from src.training.pretrain_mode import PretrainMode
from src.conf import Config

import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: Config):
    TrainingPipeline(cfg, PretrainMode()).run()


if __name__ == "__main__":
    launch(train)
