import logging

from torch_points3d.visualization import visualizer
import torch
import numpy as np
import os

log = logging.getLogger(__name__)
class Visualizer(visualizer.Visualizer):
    def __init__(self, viz_conf, num_batches, batch_size, save_dir):
        super(Visualizer, self).__init__(viz_conf, num_batches, batch_size, save_dir)

    def get_indices(self, stage):
        """This function is responsible to calculate the indices to be saved"""
        if self._contains_indices:
            return
        stage_num_batches = getattr(self, "{}_num_batches".format(stage))
        # total_items = (stage_num_batches - 1) * self._batch_size
        total_items = stage_num_batches * self._batch_size
        if stage_num_batches > 0:
            if self._num_samples_per_epoch < 0:  # All elements should be saved.
                if stage_num_batches > 0:
                    self._indices[stage] = np.arange(total_items)
                else:
                    self._indices[stage] = None
            else:
                if self._deterministic:
                    if stage not in self._indices:
                        if self._num_samples_per_epoch > total_items:
                            log.warn("Number of samples to save is higher than the number of available elements")
                        self._indices[stage] = np.random.permutation(total_items)[: self._num_samples_per_epoch]
                else:
                    if self._num_samples_per_epoch > total_items:
                        log.warn("Number of samples to save is higher than the number of available elements")
                    self._indices[stage] = np.random.permutation(total_items)[: self._num_samples_per_epoch]

    def _extract_from_PYG_npy(self, item, pos_idx):
        num_samples = item.batch.shape[0]
        batch_mask = item.batch == pos_idx
        out_data = {'name': item['name'][pos_idx]}
        for k in item.keys:
            if torch.is_tensor(item[k]) and k == 'pred':
                if item[k].shape[0] == num_samples:
                    out_data[k] = item[k][batch_mask].detach().cpu().numpy()
        return out_data

    def _extract_from_dense_npy(self, item, pos_idx):
        assert (
                item.y.shape[0] == item.pos.shape[0]
        ), "y and pos should have the same number of samples. Something is probably wrong with your data to visualise"
        num_samples = item.y.shape[0]
        out_data = {'name': item['name'][pos_idx]}
        for k in item.keys:
            if torch.is_tensor(item[k]) and k == 'pred':
                if item[k].shape[0] == num_samples:
                    out_data[k] = item[k][pos_idx].detach().cpu().numpy()
        return out_data

    def save_visuals_npy(self, visuals):
        """This function is responsible to save the data into .npy files
            Parameters:
                visuals (Dict[Data(pos=torch.Tensor, ...)]) -- Contains a dictionnary of tensors
            Make sure the saved_keys  within the config maps to the Data attributes.
        """
        if self._stage in self._indices:
            batch_indices = self._indices[self._stage] // self._batch_size
            pos_indices = self._indices[self._stage] % self._batch_size
            for idx in np.argwhere(self._seen_batch == batch_indices).flatten():
                pos_idx = pos_indices[idx]
                for visual_name, item in visuals.items():
                    if hasattr(item, "batch") and item.batch is not None:  # The PYG dataloader has been used
                        out_item = self._extract_from_PYG_npy(item, pos_idx)
                    else:
                        out_item = self._extract_from_dense_npy(item, pos_idx)

                    out_item['pred'] = out_item['pred'].astype(np.uint8)
                    dir_path = os.path.join(self._viz_path, str(self._current_epoch), self._stage)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)

                    if not os.path.exists(os.path.join(dir_path, out_item['name'])):
                        os.makedirs(os.path.join(dir_path, out_item['name']))

                    filename = "{}/{}.npy".format(out_item['name'], 'pred')
                    path_out = os.path.join(dir_path, filename)
                    np.save(path_out, out_item['pred'])
            self._seen_batch += 1
