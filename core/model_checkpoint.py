from omegaconf import DictConfig, OmegaConf
from torch_points3d.metrics import model_checkpoint

from core.model_factory import instantiate_model

import copy


class ModelCheckpoint(model_checkpoint.ModelCheckpoint):
    def __init__(self,
                 load_dir: str,
                 check_name: str,
                 selection_stage: str,
                 run_config: DictConfig = DictConfig({}),
                 resume=False,
                 strict=False, ):
        super(ModelCheckpoint, self).__init__(load_dir, check_name, selection_stage, run_config, resume,
                                              strict)

    def create_model(self, dataset, weight_name=model_checkpoint.Checkpoint._LATEST):
        if not self.is_empty:
            run_config = copy.deepcopy(self._checkpoint.run_config)
            model = instantiate_model(OmegaConf.create(run_config), dataset)
            if hasattr(self._checkpoint, "model_props"):
                for k, v in self._checkpoint.model_props.items():
                    setattr(model, k, v)
                delattr(self._checkpoint, "model_props")
            self._initialize_model(model, weight_name)
            return model
        else:
            raise ValueError("Checkpoint is empty")
