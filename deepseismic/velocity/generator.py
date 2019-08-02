from typing import Optional, Tuple

import numpy as np


class Generator(object):
    def __init__(
        self, shape: Tuple[int, ...], dtype: Optional[type] = np.float32
    ):
        self.shape = shape
        self.dtype = dtype

    def generate(self, seed: Optional[int]) -> np.ndarray:
        raise NotImplementedError
