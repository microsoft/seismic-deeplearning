from itertools import islice

import click
import h5py

from ..velocity import RoethTarantolaGenerator


@click.group()
@click.argument("output", type=click.Path())
@click.option("--append/--no-append", default=False)
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
@click.option(
    "-nz", type=int, help="Number of grid points along the third dimension"
)
@click.option("-s", "--seed", default=42, type=int, help="Random seed")
@click.pass_context
def vp(
    ctx,
    append: bool,
    n: int,
    nx: int,
    ny: int,
    nz: int,
    output: str,
    seed: int,
):
    """Vp simulation"""
    shape = (nx, ny)
    if nz is not None:
        shape += (nz,)
    ctx.obj["n"] = n
    ctx.obj["output"] = h5py.File(output, mode=("a" if append else "w"))
    ctx.obj["seed"] = seed
    ctx.obj["shape"] = shape


@vp.command()
@click.option("-l", "--layers", default=8, type=int, help="Number of layers")
@click.option(
    "--initial-vp",
    default=(1350.0, 1650.0),
    type=(float, float),
    help="Initial Vp",
)
@click.option(
    "--vp-perturbation",
    default=(-190.0, 570.0),
    type=(float, float),
    help="Per-layer Vp perturbation",
)
@click.pass_context
def rt(ctx, layers, initial_vp, vp_perturbation):
    """Röth-Tarantola model"""
    model = RoethTarantolaGenerator(
        shape=ctx.obj["shape"],
        seed=ctx.obj["seed"],
        nlayers=layers,
        initial_vp=initial_vp,
        vp_perturbation=vp_perturbation,
    )
    for i, data in enumerate(islice(model.generate_many(), ctx.obj["n"])):
        ctx.obj["output"].create_dataset(str(i), data=data, compression="gzip")
