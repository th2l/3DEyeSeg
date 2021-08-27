import numpy as np
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker


class OpenEds2021MetricTracker(SegmentationTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False,
                 ignore_label: int = IGNORE_LABEL):
        super(OpenEds2021MetricTracker, self).__init__(dataset, stage, wandb_log, use_tensorboard, ignore_label)

    def _compute_metrics(self, outputs, labels):
        mask = labels != self._ignore_label
        outputs = outputs[mask]
        labels = labels[mask]

        outputs = self._convert(outputs)
        labels = self._convert(labels)

        if len(labels) == 0:
            return

        assert outputs.shape[0] == len(labels)
        self._confusion_matrix.count_predicted_batch(labels, np.argmax(outputs, 1))

        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        # self._miou = 100 * self._confusion_matrix.get_average_intersection_union()
        self._miou_per_class = {
            i: "{:.2f}".format(100 * v)
            for i, v in enumerate(self._confusion_matrix.get_intersection_union_per_class()[0])
        }
        _ = self._miou_per_class.pop(self._ignore_label, 0)
        self._miou = np.mean([float(x) for x in list(self._miou_per_class.values())])
