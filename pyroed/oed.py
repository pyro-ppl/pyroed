import warnings

import pyro.poutine as poutine
import torch

from .inference import fit_svi
from .optimizers import optimize_simulated_annealing


def thompson_sample(
    schema,
    constraints,
    gibbs_blocks,
    response_fn,
    constraint_fn,
    model,
    dataset,
    *,
    design_size=10,
    num_svi_steps=201,
    num_sa_steps=1000,
    max_tries=1000,
    log_every=100,
    guide_temperature=4.0,
):
    """
    This trains a guide (i.e. do variational inference), draws thompson
    samples, and finds candidate designs via simulated annealing.
    """
    hot_model = poutine.scale(model, 1 / guide_temperature)

    # TODO allow swapping in MCMC here.
    guide = fit_svi(hot_model, dataset, num_steps=num_svi_steps, plot=False)

    with torch.no_grad():
        old_design = set(map(tuple, dataset["experiment_sequences"].tolist()))
        design = set()
        extent = (
            dataset["experiment_response"].max() - dataset["experiment_response"].min()
        )
        temperature_schedule = extent * torch.logspace(0.0, -2.0, num_sa_steps)
        for i in range(max_tries):
            print(".", end="", flush=True)
            with poutine.condition(data=guide()):
                coefs = model(**dataset)
            seq = optimize_simulated_annealing(
                schema,
                constraints,
                gibbs_blocks,
                response_fn,
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
