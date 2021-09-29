import warnings

import pyro.poutine as poutine
import torch

from .inference import fit_svi
from .models import model
from .optimizers import optimize_simulated_annealing


def thompson_sample(
    schema,
    constraints,
    features,
    gibbs_blocks,
    experiment,
    *,
    design_size=10,
    num_svi_steps=201,
    num_sa_steps=1000,
    max_tries=1000,
    log_every=100,
    thompson_temperature=4.0,
):
    """
    This trains a guide (i.e. do variational inference), draws thompson
    samples, and finds candidate designs via simulated annealing.
    """
    hot_model = poutine.scale(model, 1 / thompson_temperature)

    # TODO allow swapping in MCMC here.
    guide = fit_svi(
        hot_model,
        experiment,
        num_steps=num_svi_steps,
        plot=False,
    )

    with torch.no_grad():
        old_design = set(map(tuple, experiment["sequences"].tolist()))
        design = set()
        extent = experiment["response"].max() - experiment["response"].min()
        temperature_schedule = extent * torch.logspace(0.0, -2.0, num_sa_steps)
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
