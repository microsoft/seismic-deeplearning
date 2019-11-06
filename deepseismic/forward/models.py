from typing import Optional, Tuple, Union

import numpy as np
from devito import (
    Constant,
    Eq,
    Function,
    Grid,
    Operator,
    SubDomain,
    TimeFunction,
    logger,
    solve,
)

from .sources import PointSource
from .subdomains import PhysicalDomain
from .time import TimeAxis
from .types import Kernel

logger.set_log_level("WARNING")


class Model(object):
    def __init__(
        self,
        shape: Tuple[int, ...],
        origin: Tuple[float, ...],
        spacing: Tuple[float, ...],
        n_pml: Optional[int] = 0,
        dtype: Optional[type] = np.float32,
        subdomains: Optional[Tuple[SubDomain]] = (),
    ):
        shape = tuple(int(x) for x in shape)
        origin = tuple(dtype(x) for x in origin)
        n_pml = int(n_pml)
        subdomains = tuple(subdomains) + (PhysicalDomain(n_pml),)
        shape_pml = tuple(x + 2 * n_pml for x in shape)
        extent_pml = tuple(s * (d - 1) for s, d in zip(spacing, shape_pml))
        origin_pml = tuple(
            dtype(o - s * n_pml) for o, s in zip(origin, spacing)
        )
        self.grid = Grid(
            shape=shape_pml,
            extent=extent_pml,
            origin=origin_pml,
            dtype=dtype,
            subdomains=subdomains,
        )
        self.n_pml = n_pml
        self.pml = Function(name="pml", grid=self.grid)
        pml_data = np.pad(
            np.zeros(shape, dtype=dtype),
            [(n_pml,) * 2 for _ in range(self.pml.ndim)],
            mode="edge",
        )
        pml_coef = 1.5 * np.log(1000.0) / 40.0
        for d in range(self.pml.ndim):
            for i in range(n_pml):
                pos = np.abs((n_pml - i + 1) / n_pml)
                val = pml_coef * (pos - np.sin(2 * np.pi * pos) / (2 * np.pi))
                idx = [slice(0, x) for x in pml_data.shape]
                idx[d] = slice(i, i + 1)
                pml_data[tuple(idx)] += val / self.grid.spacing[d]
                idx[d] = slice(
                    pml_data.shape[d] - i, pml_data.shape[d] - i + 1
                )
                pml_data[tuple(idx)] += val / self.grid.spacing[d]
        pml_data = np.pad(
            pml_data,
            [(i.left, i.right) for i in self.pml._size_halo],
            mode="edge",
        )
        self.pml.data_with_halo[:] = pml_data
        self.shape = shape

    @property
    def domain_size(self) -> Tuple[float, ...]:
        return tuple((d - 1) * s for d, s in zip(self.shape, self.spacing))

    @property
    def dtype(self) -> type:
        return self.grid.dtype

    @property
    def spacing(self):
        return self.grid.spacing

    @property
    def spacing_map(self):
        return self.grid.spacing_map

    @property
    def time_spacing(self):
        return self.grid.stepping_dim.spacing


class VelocityModel(Model):
    def __init__(
        self,
        shape: Tuple[int, ...],
        origin: Tuple[float, ...],
        spacing: Tuple[float, ...],
        vp: Union[float, np.ndarray],
        space_order: Optional[int] = None,
        n_pml: Optional[int] = 0,
        dtype: Optional[type] = np.float32,
        subdomains: Optional[Tuple[SubDomain]] = (),
    ):
        super().__init__(shape, origin, spacing, n_pml, dtype, subdomains)
        if isinstance(vp, np.ndarray):
            assert space_order is not None
            self.m = Function(
                name="m", grid=self.grid, space_order=int(space_order)
            )
        else:
            self.m = Constant(name="m", value=1.0 / float(vp) ** 2.0)
        self.vp = vp

    @property
    def vp(self) -> Union[float, np.ndarray]:
        return self._vp

    @vp.setter
    def vp(self, vp: Union[float, np.ndarray]) -> None:
        self._vp = vp
        if isinstance(vp, np.ndarray):
            pad_widths = [
                (self.n_pml + i.left, self.n_pml + i.right)
                for i in self.m._size_halo
            ]
            self.m.data_with_halo[:] = np.pad(
                1.0 / self.vp ** 2.0, pad_widths, mode="edge"
            )
        else:
            self.m.data = 1.0 / float(vp) ** 2.0

    def solve(
        self,
        source: PointSource,
        receivers: PointSource,
        time_range: TimeAxis,
        space_order: Optional[int] = 4,
        kernel: Optional[Kernel] = Kernel.OT2,
    ) -> np.ndarray:
        assert isinstance(kernel, Kernel)
        u = TimeFunction(
            name="u", grid=self.grid, time_order=2, space_order=space_order
        )
        H = u.laplace
        if kernel is Kernel.OT4:
            H += self.time_spacing ** 2 / 12 * u.laplace2(1 / self.m)
        eq = Eq(
            u.forward, solve(self.m * u.dt2 - H + self.pml * u.dt, u.forward)
        )
        src_term = source.inject(
            field=u.forward, expr=source * self.time_spacing ** 2 / self.m
        )
        rec_term = receivers.interpolate(expr=u)
        op = Operator([eq] + src_term + rec_term, subs=self.spacing_map)
        op(time=time_range.num - 1, dt=time_range.step)
        return receivers.data
