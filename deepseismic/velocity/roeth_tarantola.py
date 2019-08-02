from typing import Optional, Tuple

import numpy as np

from .generator import Generator


class RoethTarantolaGenerator(Generator):
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[type] = np.float32,
        seed: Optional[int] = None,
        horizontal_dim: Optional[int] = -1,
        nlayers: Optional[int] = 8,
        initial_vp: Optional[Tuple[float, float]] = (1350.0, 1650.0),
        vp_perturbation: Optional[Tuple[float, float]] = (-190.0, 570.0),
    ):
        super().__init__(shape, dtype, seed)
        self.horizontal_dim = horizontal_dim
        self.nlayers = nlayers
        self.initial_vp = initial_vp
        self.vp_perturbation = vp_perturbation

    def generate(self) -> np.ndarray:
        vp = np.zeros(self.shape, dtype=self.dtype)
        dim = self.horizontal_dim
        layer_idx = np.round(
            np.linspace(0, self.shape[dim], self.nlayers + 1)
        ).astype(np.int)
        vp_idx = [slice(0, x) for x in vp.shape]
        layer_vp = None
        for i in range(self.nlayers):
            vp_idx[dim] = slice(layer_idx[i], layer_idx[i + 1])
            layer_vp = (
                self._prng.uniform(*self.initial_vp)
                if layer_vp is None
                else layer_vp + self._prng.uniform(*self.vp_perturbation)
            )
            vp[tuple(vp_idx)] = layer_vp
        return vp
