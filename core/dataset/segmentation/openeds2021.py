"""
Original source: https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d/datasets/segmentation/shapenet.py
Latest commit 8972080 on Jul 29, 2020
"""
import os
import os.path as osp
import shutil
import json

# from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from .metric_tracker import OpenEds2021MetricTracker
from tqdm.auto import tqdm as tq
from itertools import repeat, product
import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import read_ply
from torch_points3d.datasets.base_dataset import BaseDataset


class OpenEDS3D(InMemoryDataset):
    """
    Class to handle OpenEDS2021 3D Eye segmentation
    """
    num_classes = 5  # the pupil, the iris, the sclera, the eyelashes and the skin (background)

    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        super(OpenEDS3D, self).__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            path_idx = 0
        elif split == 'val':
            path_idx = 1
        else:
            path_idx = 2  # test

        self.data, self.slices = torch.load(self.processed_paths[path_idx])

    @property
    def raw_file_names(self):
        """
        A list of files in the raw_dir which needs to be found in order to skip the download.
        :return:
        """
        return ['train', 'val', 'test']

    @property
    def processed_file_names(self):
        """
        A list of files in the processed_dir which needs to be found in order to skip the processing.
        :return:
        """
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        """
        Processes raw data and saves it into the processed_dir.
        :return:
        """
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('val'), self.processed_paths[1])
        torch.save(self.process_set('test'), self.processed_paths[2])
        pass

    def process_set(self, dataset):
        folder = osp.join(self.raw_dir, dataset)
        paths = np.loadtxt('{}/../splits/{}.txt'.format(self.raw_dir, dataset), dtype=str)
        data_list = []
        data_length = []

        for path in paths:
            data = read_ply(osp.join(folder, path, 'pointcloud.ply'))
            if dataset != 'test':
                label = np.load(osp.join(folder, path, 'labels.npy'))
                data.y = torch.from_numpy(label).long()
                data_length.append(len(label))

                # if path == '0020_pose_16':
                #     print('original: ', path, data.y.shape)

            data.name = path
            data_list.append(data)

        # print(dataset, np.unique(data_length, return_counts=True))
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)


class OpenEDS3DDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super(OpenEDS3DDataset, self).__init__(dataset_opt)
        self._data_path = self._data_path.replace('/openeds3d', '')
        self.train_dataset = OpenEDS3D(self._data_path, split='train', pre_transform=self.pre_transform,
                                       transform=self.train_transform)
        self.val_dataset = OpenEDS3D(self._data_path, split='val', pre_transform=self.pre_transform,
                                     transform=self.val_transform)
        # tmp = self.val_dataset[3]
        self.test_dataset = OpenEDS3D(self._data_path, split='test', pre_transform=self.pre_transform,
                                      transform=self.test_transform)

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        # 0 - pupil, 1 - iris, 2 - sclera, 3 - eyelashes, 4 = background
        return OpenEds2021MetricTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, ignore_label=4)