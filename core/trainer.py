import logging
import copy
import os

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.models.base_model import BaseModel

from torch_points3d.trainer import Trainer
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.visualization import visualizer

# Utils import
from torch_points3d.utils.colors import COLORS
from torch_points3d.utils.wandb_utils import Wandb
import torch

from .dataset_factory import instantiate_dataset
from .model_factory import instantiate_model
from .model_checkpoint import ModelCheckpoint
from .visualizer import Visualizer

log = logging.getLogger(__name__)


class Eye3DTrainer(Trainer):
    def __init__(self, cfg):
        self._cfg = cfg
        self._initialize_trainer()

    def _initialize_trainer(self):
        # Enable CUDNN BACKEND

        if not self.has_training:
            self._cfg.training = self._cfg
            resume = bool(self._cfg.checkpoint_dir)
        else:
            resume = bool(self._cfg.training.checkpoint_dir)

        torch.backends.cudnn.enabled = self.enable_cudnn
        # Get device
        if self._cfg.training.cuda > -1 and torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(self._cfg.training.cuda)
        else:
            device = "cpu"
        self._device = torch.device(device)
        log.info("DEVICE : {}".format(self._device))

        # Profiling
        if self.profiling:
            # Set the num_workers as torch.utils.bottleneck doesn't work well with it
            self._cfg.training.num_workers = 0

        # Start Wandb if public
        if self.wandb_log:
            Wandb.launch(self._cfg, self._cfg.wandb.public and self.wandb_log)

        # Checkpoint

        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            self._cfg.training.checkpoint_dir,
            self._cfg.model_name,
            self._cfg.training.weight_name,
            run_config=self._cfg,
            resume=resume,
        )

        # Create model and datasets
        if not self._checkpoint.is_empty:
            self._dataset: BaseDataset = instantiate_dataset(self._checkpoint.data_config)
            self._model: BaseModel = self._checkpoint.create_model(
                self._dataset, weight_name=self._cfg.training.weight_name
            )
        else:
            self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
            self._model: BaseModel = instantiate_model(copy.deepcopy(self._cfg), self._dataset)
            self._model.instantiate_optimizers(self._cfg, "cuda" in device)
            self._model.set_pretrained_weights()
            if not self._checkpoint.validate(self._dataset.used_properties):
                log.warning(
                    "The model will not be able to be used from pretrained weights without the corresponding dataset. Current properties are {}".format(
                        self._dataset.used_properties
                    )
                )
        self._checkpoint.dataset_properties = self._dataset.used_properties

        log.info(self._model)

        self._model.log_optimizers()
        log.info("Model size = %i", sum(param.numel() for param in self._model.parameters() if param.requires_grad))

        # Set dataloaders
        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            self._cfg.training.shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
        )
        log.info(self._dataset)

        # Verify attributes in dataset
        self._model.verify_data(self._dataset.train_dataset[0])

        print('Use Tensorboard: ', self.tensorboard_log)
        # Choose selection stage
        selection_stage = getattr(self._cfg, "selection_stage", "")
        self._checkpoint.selection_stage = self._dataset.resolve_saving_stage(selection_stage)
        self._tracker: BaseTracker = self._dataset.get_tracker(self.wandb_log, self.tensorboard_log)

        if self.wandb_log:
            Wandb.launch(self._cfg, not self._cfg.wandb.public and self.wandb_log)

        # Run training / evaluation
        self._model = self._model.to(self._device)
        if self.has_visualization:
            self._visualizer = Visualizer(
                self._cfg.visualization, self._dataset.num_batches, self._dataset.batch_size, os.getcwd()
            )

    def _test_epoch(self, epoch, stage_name: str):
        voting_runs = self._cfg.get("voting_runs", 1)
        if stage_name == "test":
            loaders = self._dataset.test_dataloaders
        else:
            loaders = [self._dataset.val_dataloader]

        self._model.eval()
        if self.enable_dropout:
            self._model.enable_dropout_in_eval()

        for loader in loaders:
            stage_name = loader.dataset.name
            self._tracker.reset(stage_name)
            if self.has_visualization:
                self._visualizer.reset(epoch, stage_name)
            if not self._dataset.has_labels(stage_name) and not self.tracker_options.get(
                    "make_submission", False
            ):  # No label, no submission -> do nothing
                log.warning("No forward will be run on dataset %s." % stage_name)
                continue

            for i in range(voting_runs):
                with Ctq(loader) as tq_loader:
                    for data in tq_loader:
                        with torch.no_grad():
                            self._model.set_input(data, self._device)
                            self._model.forward(epoch=epoch)
                            self._tracker.track(self._model, data=data, **self.tracker_options)
                        tq_loader.set_postfix(**self._tracker.get_metrics(), color=COLORS.TEST_COLOR)

                        if self.has_visualization and self._visualizer.is_active:
                            self._visualizer.save_visuals_npy(self._model.get_current_visuals())
                            # self._visualizer.save_visuals(self._model.get_current_visuals())

                        if self.early_break:
                            break

                        if self.profiling:
                            if i > self.num_batches:
                                return 0

            self._finalize_epoch(epoch)
            self._tracker.print_summary()

    @property
    def has_tensorboard(self):
        return getattr(self._cfg.training, "tensorboard", False)

    @property
    def tensorboard_log(self):
        if self.has_tensorboard:
            return getattr(self._cfg.training.tensorboard, "log", False)
        else:
            return False
