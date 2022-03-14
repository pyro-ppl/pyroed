import warnings
from typing import Dict, Set, Tuple

import pyro.poutine as poutine
import torch

from .inference import fit_mcmc, fit_svi
from .models import model
from .optimizers import optimize_simulated_annealing
from .typing import Blocks, Constraints, Schema


def thompson_sample(
    schema: Schema,
    constraints: Constraints,
    features: Blocks,
    gibbs_blocks: Blocks,
    experiment: Dict[str, torch.Tensor],
    *,
    design_size=10,
    inference="svi",
    mcmc_num_samples=500,
    mcmc_warmup_steps=500,
    mcmc_num_chains=1,
    svi_num_steps=201,
    sa_num_steps=1000,
    max_tries=1000,
    thompson_temperature=1.0,
    jit_compile=False,
    log_every=100,
) -> Set[Tuple[int, ...]]:
    """
    This trains a guide (i.e. do variational inference), draws thompson
    samples, and finds candidate designs via simulated annealing.
    """

    @poutine.scale(scale=1 / thompson_temperature)
    def hot_model():
        return model(schema, features, experiment)

    # Fit a posterior distribution over parameters given experiment data.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "torch.tensor results are registered as constants",
            torch.jit.TracerWarning,
        )
        if inference == "svi":
            sampler = fit_svi(
                hot_model,
                num_steps=svi_num_steps,
                jit_compile=jit_compile,
                plot=False,
                log_every=log_every,
            )
        elif inference == "mcmc":
            sampler = fit_mcmc(
                hot_model,
                num_samples=mcmc_num_samples,
                warmup_steps=mcmc_warmup_steps,
                num_chains=mcmc_num_chains,
                jit_compile=jit_compile,
            )
        else:
            raise ValueError(f"Unknown inference type: {inference}")

    # Repeatedly sample coefficients from the posterior,
    # and for each sample find an optimal sequence.
    with torch.no_grad():
        logits = experiment["response"].clamp(min=0.001, max=0.999).logit()
        extent = logits.max() - logits.min()
        temperature_schedule = extent * torch.logspace(0.0, -2.0, sa_num_steps)

        old_design = set(map(tuple, experiment["sequences"].tolist()))
        design: Set[Tuple[int, ...]] = set()

        for i in range(max_tries):
            if log_every:
                print(".", end="", flush=True)
            with poutine.condition(data=sampler()):
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
