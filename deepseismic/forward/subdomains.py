from typing import Dict, Iterable, Tuple

from devito import Dimension, SubDomain


class PhysicalDomain(SubDomain):

    name = "physical_domain"

    def __init__(self, npml: int):
        super().__init__()
        self.npml = npml

    def define(
        self, dimensions: Iterable[Dimension]
    ) -> Dict[Dimension, Tuple[str, int, int]]:
        return {d: ("middle", self.npml, self.npml) for d in dimensions}
