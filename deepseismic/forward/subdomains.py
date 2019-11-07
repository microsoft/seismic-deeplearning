# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Iterable, Tuple

from devito import Dimension, SubDomain


class PhysicalDomain(SubDomain):
    name = "physical_domain"

    def __init__(self, n_pml: int):
        super().__init__()
        self.n_pml = n_pml

    def define(
        self, dimensions: Iterable[Dimension]
    ) -> Dict[Dimension, Tuple[str, int, int]]:
        return {d: ("middle", self.n_pml, self.n_pml) for d in dimensions}
