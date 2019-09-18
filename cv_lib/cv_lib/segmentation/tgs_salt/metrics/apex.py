import torch.distributed as dist
from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
import torch

@torch.no_grad()
def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

@torch.no_grad()
def gather_tensor(tensor, world_size):
    gather_t = [torch.ones_like(tensor).cuda() for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return gather_t


class AverageMetric(Metric):
    def __init__(self, world_size, batch_size, output_transform=lambda x: x):
        super(AverageMetric, self).__init__(output_transform=output_transform)
        self._world_size = world_size
        self._batch_size = batch_size
        self._metric_name = "Metric"

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @torch.no_grad()
    def update(self, output):
        reduced_metric = reduce_tensor(output, self._world_size)
        self._sum += reduced_metric * self._batch_size
        self._num_examples += self._batch_size

    @torch.no_grad()
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                f"{self._metric_name} must have at least one example before it can be computed."
            )
        return self._sum / self._num_examples


class LossMetric(AverageMetric):
    def __init__(self, loss_fn, world_size, batch_size, output_transform=lambda x: x):
        super(LossMetric, self).__init__(
            world_size, batch_size, output_transform=output_transform
        )
        self._loss_fn = loss_fn
        self._metric_name = "Loss"

    def update(self, output):
        pred, y = output
        loss = self._loss_fn(pred, y)
        super().update(loss)


class KaggleMetric(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(KaggleMetric, self).__init__(output_transform=output_transform)

    @torch.no_grad()
    def reset(self):
        self._predictions = torch.tensor([], dtype=torch.float32).cuda()
        self._targets = torch.tensor([], dtype=torch.float32).cuda()

    @torch.no_grad()
    def update(self, output):
        y_pred, y = output
        y_pred = y_pred.type_as(self._predictions)
        y = y.type_as(self._targets)
        self._predictions = torch.cat([self._predictions, y_pred], dim=0)
        self._targets = torch.cat([self._targets, y], dim=0)
    
    @torch.no_grad()
    def compute(self):
        gather_predictions = gather_tensor(self._predictions, self._world_size)
        gather_targets = gather_tensor(self._targets, self._world_size)
        predictions = torch.cat(gather_predictions, dim=0)
        targets = torch.cat(gather_targets, dim=0)
        precision, _, _ = do_kaggle_metric(predictions.detach().cpu().numpy(), targets.detach().cpu().numpy(), 0.5)
        precision = precision.mean()
        return precision



class PixelwiseAccuracyMetric(AverageMetric):
    def __init__(self, world_size, batch_size, output_transform=lambda x: x, threshold=0.5):
        super(PixelwiseAccuracyMetric, self).__init__(
            world_size, batch_size, output_transform=output_transform
        )
        self._acc_fn = pixel_wise_accuracy2
        self._metric_name = "PixelwiseAccuracy"
        self._threshold = threshold

    def update(self, output):
        y_pred, y = output
        y_pred[y_pred>self._threshold] = 1
        y_pred[y_pred<=self._threshold] = 0
        super().update(self._acc_fn(y_pred, y))
