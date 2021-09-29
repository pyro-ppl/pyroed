import warnings
from typing import Set, Tuple

import pyro.poutine as poutine
import torch

from .inference import fit_svi
from .models import model
from .optimizers import optimize_simulated_annealing
from .typing import Constraints, Features, GibbsBlocks, Schema


def thompson_sample(
    schema: Schema,
    constraints: Constraints,
    features: Features,
    gibbs_blocks: GibbsBlocks,
    experiment,
    *,
    design_size=10,
    num_svi_steps=201,
    num_sa_steps=1000,
    max_tries=1000,
    thompson_temperature=4.0,
    log_every=100,
) -> Set[Tuple[int, ...]]:
    """
    This trains a guide (i.e. do variational inference), draws thompson
    samples, and finds candidate designs via simulated annealing.
    """

    @poutine.scale(scale=1 / thompson_temperature)
    def hot_model():
        return model(schema, features, experiment)

    # TODO allow swapping in MCMC here.
    guide = fit_svi(
        hot_model,
        num_steps=num_svi_steps,
        plot=False,
    )

    with torch.no_grad():
        logits = experiment["response"].clamp(min=0.005, max=0.995).logit()
        extent = logits.max() - logits.min()
        temperature_schedule = extent * torch.logspace(0.0, -2.0, num_sa_steps)

        old_design = set(map(tuple, experiment["sequences"].tolist()))
        design: Set[Tuple[int, ...]] = set()

        for i in range(max_tries):
            print(".", end="", flush=True)
            with poutine.condition(data=guide()):
                coefs = model(schema, features, experiment)

            seq = optimize_simulated_annealing(
                schema,
                constraints,
                features,
                gibbs_blocks,
                coefs,
                temperature_schedule=temperature_schedule,
                log_every=log_every,
            )

            seq = tuple(seq.tolist())
            if seq not in old_design:
                design.add(seq)
            if len(design) >= design_size:
                break

    if len(design) < design_size:
        warnings.warn(f"Found design of only {len(design)}/{design_size} sequences")

    return design
