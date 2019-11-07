# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional

import numpy as np


class TimeAxis(object):
    def __init__(
        self,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        num: Optional[int] = None,
        step: Optional[float] = None,
        dtype: Optional[type] = np.float32,
    ):
        if start is None:
            start = step * (1 - num) + stop
        elif stop is None:
            stop = step * (num - 1) + start
        elif num is None:
            num = int(np.ceil((stop - start + step) / step))
            stop = step * (num - 1) + start
        elif step is None:
            step = (stop - start) / (num - 1)
        else:
            raise ValueError
        self.start = start
        self.stop = stop
        self.num = num
        self.step = step
        self.dtype = dtype

    @property
    def time_values(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, self.num, dtype=self.dtype)
