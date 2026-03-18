import sys

sys.path.insert(0, ".")

from src.training import TrainingPipeline, launch
from src.training.finetune_mode import FinetuneMode
from src.conf import Config

import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: Config):
    TrainingPipeline(cfg, FinetuneMode()).run()


if __name__ == "__main__":
    launch(train)
