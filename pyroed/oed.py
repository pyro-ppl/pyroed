import warnings
from typing import Dict, Optional, Set, Tuple

import pyro.poutine as poutine
import torch

from .inference import fit_mcmc, fit_svi
from .models import model
from .optimizers import optimize_simulated_annealing
from .typing import Blocks, Constraints, Schema


def thompson_sample(
    schema: Schema,
    constraints: Constraints,
    feature_blocks: Blocks,
    gibbs_blocks: Blocks,
    experiment: Dict[str, torch.Tensor],
    *,
    design_size: int = 10,
    response_type: str = "unit_interval",
    inference: str = "svi",
    mcmc_num_samples: int = 500,
    mcmc_warmup_steps: int = 500,
    mcmc_num_chains: int = 1,
    svi_num_steps: int = 201,
    sa_num_steps: int = 1000,
    max_tries: int = 1000,
    thompson_temperature: float = 1.0,
    jit_compile: Optional[bool] = None,
    log_every: int = 100,
) -> Set[Tuple[int, ...]]:
    """
    This trains a guide (i.e. do variational inference), draws thompson
    samples, and finds candidate designs via simulated annealing.

    :param OrderedDict schema: A schema dict.
    :param list constraints: A list of constraints.
    :param list feature_blocks: A list of choice blocks for linear regression.
    :param list gibbs_blocks: A list of choice blocks for Gibbs sampling.
    :param dict experiment: A dict containing all old experiment data.
    :param int design_size: Number of designs to try to return (sometimes
        fewer designs are found).
    :returns: A design as a set of tuples of choices.
    :rtype: set
    """
    # Pass max_batch_id separately as a python int to allow jitting.
    max_batch_id = int(experiment["batch_ids"].max())

    @poutine.scale(scale=1 / thompson_temperature)
    def hot_model():
        return model(
            schema,
            feature_blocks,
            experiment,
            max_batch_id=max_batch_id,
            response_type=response_type,
        )

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
    with torch.no_grad(), poutine.mask(mask=False):
        logits = experiment["responses"].clamp(min=0.001, max=0.999).logit()
        extent = logits.max() - logits.min()
        temperature_schedule = extent * torch.logspace(0.0, -2.0, sa_num_steps)

        old_design = set(map(tuple, experiment["sequences"].tolist()))
        design: Set[Tuple[int, ...]] = set()

        for i in range(max_tries):
            if log_every:
                print(".", end="", flush=True)
            coefs = poutine.condition(hot_model, sampler())()

            seq = optimize_simulated_annealing(
                schema,
                constraints,
                gibbs_blocks,
                coefs,
                temperature_schedule=temperature_schedule,
                log_every=log_every,
            )

            new_seq = tuple(seq.tolist())
            if new_seq not in old_design:
                design.add(new_seq)
            if len(design) >= design_size:
                break

    if len(design) < design_size:
        warnings.warn(f"Found design of only {len(design)}/{design_size} sequences")

    return design
