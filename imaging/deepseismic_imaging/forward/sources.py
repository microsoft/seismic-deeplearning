from typing import Optional

import numpy as np
import sympy
from devito.types import Dimension, SparseTimeFunction
from devito.types.basic import _SymbolCache
from scipy import interpolate

from .time import TimeAxis


class PointSource(SparseTimeFunction):
    def __new__(cls, *args, **kwargs):
        if cls in _SymbolCache:
            options = kwargs.get("options", {})
            obj = sympy.Function.__new__(cls, *args, **options)
            obj._cached_init()
            return obj
        name = kwargs.pop("name")
        grid = kwargs.pop("grid")
        time_range = kwargs.pop("time_range")
        time_order = kwargs.pop("time_order", 2)
        p_dim = kwargs.pop("dimension", Dimension(name="p_%s" % name))
        npoint = kwargs.pop("npoint", None)
        coordinates = kwargs.pop(
            "coordinates", kwargs.pop("coordinates_data", None)
        )
        if npoint is None:
            assert (
                coordinates is not None
            ), "Either `npoint` or `coordinates` must be provided"
            npoint = coordinates.shape[0]
        obj = SparseTimeFunction.__new__(
            cls,
            name=name,
            grid=grid,
            dimensions=(grid.time_dim, p_dim),
            npoint=npoint,
            nt=time_range.num,
            time_order=time_order,
            coordinates=coordinates,
            **kwargs
        )
        obj._time_range = time_range
        data = kwargs.get("data")
        if data is not None:
            obj.data[:] = data
        return obj

    @property
    def time_range(self) -> TimeAxis:
        return self._time_range

    @property
    def time_values(self) -> np.ndarray:
        return self._time_range.time_values

    def resample(
        self,
        dt: Optional[float] = None,
        num: Optional[int] = None,
        rtol: Optional[float] = 1.0e-5,
        order: Optional[int] = 3,
    ):
        assert (dt is not None) ^ (
            num is not None
        ), "Exactly one of `dt` or `num` must be provided"
        start = self._time_range.start
        stop = self._time_range.stop
        dt0 = self._time_range.step
        if dt is not None:
            new_time_range = TimeAxis(start=start, stop=stop, step=dt)
        else:
            new_time_range = TimeAxis(start=start, stop=stop, num=num)
            dt = new_time_range.step
        if np.isclose(dt0, dt, rtol=rtol):
            return self
        n_traces = self.data.shape[1]
        new_traces = np.zeros(
            (new_time_range.num, n_traces), dtype=self.data.dtype
        )
        for j in range(n_traces):
            tck = interpolate.splrep(
                self._time_range.time_values, self.data[:, j], k=order
            )
            new_traces[:, j] = interpolate.splev(
                new_time_range.time_values, tck
            )
        return PointSource(
            name=self.name,
            grid=self.grid,
            time_range=new_time_range,
            coordinates=self.coordinates.data,
            data=new_traces,
        )

    _pickle_kwargs = SparseTimeFunction._pickle_kwargs + ["time_range"]
    _pickle_kwargs.remove("nt")  # Inferred from time_range


class Receiver(PointSource):
    pass


class WaveletSource(PointSource):
    def __new__(cls, *args, **kwargs):
        if cls in _SymbolCache:
            options = kwargs.get("options", {})
            obj = sympy.Function.__new__(cls, *args, **options)
            obj._cached_init()
            return obj
        npoint = kwargs.pop("npoint", 1)
        obj = PointSource.__new__(cls, npoint=npoint, **kwargs)
        obj.f0 = kwargs.get("f0")
        for p in range(npoint):
            obj.data[:, p] = obj.wavelet(obj.f0, obj.time_values)
        return obj

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(WaveletSource, self).__init__(*args, **kwargs)

    def wavelet(self, f0: float, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    _pickle_kwargs = PointSource._pickle_kwargs + ["f0"]


class RickerSource(WaveletSource):
    def wavelet(self, f0: float, t: np.ndarray) -> np.ndarray:
        r = np.pi * f0 * (t - 1.0 / f0)
        return (1.0 - 2.0 * r ** 2.0) * np.exp(-r ** 2.0)
