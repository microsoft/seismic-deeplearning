# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from ignite.metrics import Metric
import torch
import numpy as np


def _torch_hist(label_true, label_pred, n_class):
    """Calculates the confusion matrix for the labels
    
    Args:
        label_true ([type]): [description]
        label_pred ([type]): [description]
        n_class ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    
    assert len(label_true.shape) == 1, "Labels need to be 1D"
    assert len(label_pred.shape) == 1, "Predictions need to be 1D"
    mask = (label_true >= 0) & (label_true < n_class)
    hist = torch.bincount(n_class * label_true[mask] + label_pred[mask], minlength=n_class ** 2).reshape(
        n_class, n_class
    )
    return hist


def _default_tensor(image_height, image_width, pad_value=255):
    return torch.full((image_height, image_width), pad_value, dtype=torch.long)


# TODO: make output transform unpad and scale down mask
# scale up y_pred and remove padding
# issue: https://github.com/microsoft/seismic-deeplearning/issues/276
class InlineMeanIoU(Metric):
    """Compute Mean IoU for Inline

    Notes:
        This metric collects all the patches and recomposes the predictions and masks
        into inlines. These are then used to calculate the mean IoU.
    """

    def __init__(
        self,
        image_height,
        image_width,
        patch_size,
        num_classes,
        padding=0,
        scale=1,
        pad_value=255,
        output_transform=lambda x: x,
    ):
        """Create instance of InlineMeanIoU

        Args:
            image_height (int): height of inline
            image_width (int): width of inline
            patch_size (int): patch size
            num_classes (int): number of classes in dataset
            padding (int, optional): the amount of padding to height and width,
                e.g 200 padded to 256 - padding=56. Defaults to 0
            scale (int, optional): the scale factor applied to the patch,
                e.g 100 scaled to 200 - scale=2. Defaults to 1
            pad_value (int):  the constant value used for padding Defaults to 255
            output_transform (callable, optional): a callable that is used to transform
                the ignite.engine.Engine's `process_function`'s output into the form
                expected by the metric. This can be useful if, for example, if you have
                a multi-output model and you want to compute the metric with respect to
                one of the outputs.
        """
        self._image_height = image_height
        self._image_width = image_width
        self._patch_size = patch_size
        self._pad_value = pad_value
        self._num_classes = num_classes
        self._scale = scale
        self._padding = padding
        super(InlineMeanIoU, self).__init__(output_transform=output_transform)

    def reset(self):
        self._pred_dict = defaultdict(
            lambda: _default_tensor(
                self._image_height * self._scale, self._image_width * self._scale, pad_value=self._pad_value,
            )
        )
        self._mask_dict = defaultdict(
            lambda: _default_tensor(
                self._image_height * self._scale, self._image_width * self._scale, pad_value=self._pad_value,
            )
        )

    def update(self, output):
        y_pred, y, ids, patch_locations = output
        # TODO: Make assertion exception
        # issue: https://github.com/microsoft/seismic-deeplearning/issues/276
        max_prediction = y_pred.max(1)[1].squeeze()
        assert y.shape == max_prediction.shape, "Shape not the same"

        for pred, mask, id, patch_loc in zip(max_prediction, y, ids, patch_locations):
            # ! With overlapping patches this does not aggregate the results,
            # ! it simply overwrites them
            # If patch is padded ingore padding
            pad = int(self._padding // 2)
            pred = pred[pad : pred.shape[0] - pad, pad : pred.shape[1] - pad]
            mask = mask[pad : mask.shape[0] - pad, pad : mask.shape[1] - pad]

            # Get the ares of the mask that is not padded
            # Determine the left top edge and bottom right edge
            # Use this to calculate the rectangular area that contains predictions
            non_padded_mask = torch.nonzero((mask - self._pad_value).abs())
            y_start, x_start = non_padded_mask.min(0)[0]
            y_end, x_end = non_padded_mask.max(0)[0]
            height = (y_end + 1) - y_start
            width = (x_end + 1) - x_start

            self._pred_dict[id][
                patch_loc[0] * 2 : patch_loc[0] * 2 + height, patch_loc[1] * 2 : patch_loc[1] * 2 + width,
            ] = pred[y_start : y_end + 1, x_start : x_end + 1]

            self._mask_dict[id][
                patch_loc[0] * 2 : patch_loc[0] * 2 + height, patch_loc[1] * 2 : patch_loc[1] * 2 + width,
            ] = mask[y_start : y_end + 1, x_start : x_end + 1]

    def iou_per_inline(self):
        iou_per_inline = {}
        for id in self._pred_dict:
            confusion_matrix = _torch_hist(
                torch.flatten(self._mask_dict[id]),
                torch.flatten(self._pred_dict[id]),  # Get the maximum index
                self._num_classes,
            )
            hist = confusion_matrix.cpu().numpy()
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
            iou_per_inline[id] = np.nanmean(iu)
        return iou_per_inline

    @property
    def predictions(self):
        return self._pred_dict

    @property
    def masks(self):
        return self._mask_dict

    def compute(self):
        iou_dict = self.iou_per_inline()
        return np.mean(list(iou_dict.values()))
