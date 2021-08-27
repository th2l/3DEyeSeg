import os

os.environ['OMP_NUM_THREADS'] = '8'
import hydra
from hydra.core.global_hydra import GlobalHydra

from omegaconf import OmegaConf
from core.trainer import Eye3DTrainer
import torch
import random
import numpy as np


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(cfg.pretty())

    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(0)
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    trainer = Eye3DTrainer(cfg)
    trainer.train()

    # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
