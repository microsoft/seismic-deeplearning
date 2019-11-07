# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from itertools import islice
from typing import Tuple

import click
import h5py

from ..velocity import RoethTarantolaGenerator

click.option = partial(click.option, show_default=True)


@click.group()
@click.argument("output", type=click.Path())
@click.option(
    "--append/--no-append", default=False, help="Whether to append to output file",
)
@click.option("-n", default=1, type=int, help="Number of simulations")
@click.option(
    "-nx",
    default=100,
    type=int,
    help="Number of grid points along the first dimension",
)
@click.option(
    "-ny",
    default=100,
    type=int,
    help="Number of grid points along the second dimension",
)
@click.option("-nz", type=int, help="Number of grid points along the third dimension")
@click.option("-s", "--seed", default=42, type=int, help="Random seed")
@click.pass_context
def vp(
    ctx, append: bool, n: int, nx: int, ny: int, nz: int, output: str, seed: int,
):
    """Vp simulation"""
    shape = (nx, ny)
    if nz is not None:
        shape += (nz,)
    output_file = h5py.File(output, mode=("a" if append else "w"))
    output_group = output_file.create_group(
        str(max((int(x) for x in output_file.keys()), default=-1) + 1)
    )
    ctx.obj["n"] = n
    ctx.obj["output_file"] = output_file
    ctx.obj["output_group"] = output_group
    ctx.obj["seed"] = seed
    ctx.obj["shape"] = shape


@vp.command()
@click.option("--n-layers", default=8, type=int, help="Number of layers")
@click.option(
    "--initial-vp",
    default=(1350.0, 1650.0),
    type=(float, float),
    help="Initial Vp (in km/s)",
)
@click.option(
    "--vp-perturbation",
    default=(-190.0, 570.0),
    type=(float, float),
    help="Per-layer Vp perturbation (in km/s)",
)
@click.pass_context
def rt(
    ctx,
    initial_vp: Tuple[float, float],
    n_layers: int,
    vp_perturbation: Tuple[float, float],
):
    """Röth-Tarantola model"""
    model = RoethTarantolaGenerator(
        shape=ctx.obj["shape"],
        seed=ctx.obj["seed"],
        n_layers=n_layers,
        initial_vp=initial_vp,
        vp_perturbation=vp_perturbation,
    )
    group = ctx.obj["output_group"]
    with click.progressbar(length=ctx.obj["n"]) as bar:
        for i, data in enumerate(islice(model.generate_many(), ctx.obj["n"])):
            group.create_dataset(str(i), data=data, compression="gzip")
            bar.update(1)
