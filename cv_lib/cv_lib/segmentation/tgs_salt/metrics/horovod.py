from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
import torch
import horovod.torch as hvd


def reduce_tensor(tensor):
    """Computes average of tensor
    
    Args:
        tensor ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    return hvd.allreduce(tensor)


def gather_tensor(tensor):
    return hvd.allgather(tensor)


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
        reduced_metric = reduce_tensor(output)
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
        predictions = gather_tensor(self._predictions)
        targets = gather_tensor(self._targets)
        precision, _, _ = do_kaggle_metric(predictions.detach().cpu().numpy(), targets.detach().cpu().numpy(), 0.5)
        precision = precision.mean()
        return precision

