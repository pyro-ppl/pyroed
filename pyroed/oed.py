import functools
import warnings
from typing import Callable, Dict, Optional, Set, Tuple

import pyro.poutine as poutine
import torch
from pyro.infer.reparam import AutoReparam

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
    feature_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    response_type: str = "unit_interval",
    inference: str = "svi",
    mcmc_num_samples: int = 500,
    mcmc_warmup_steps: int = 500,
    mcmc_num_chains: int = 1,
    svi_num_steps: int = 501,
    svi_reparam: bool = True,
    svi_plot: bool = False,
    sa_num_steps: int = 1000,
    max_tries: int = 1000,
    thompson_temperature: float = 1.0,
    jit_compile: Optional[bool] = None,
    log_every: int = 100,
) -> Set[Tuple[int, ...]]:
    """
    Performs Bayesian optimization via Thompson sampling.

    This fits a Bayesian model to existing experimental data, and draws
    Thompson samples wrt that model. To draw each Thompson sample, this first
    samples parameters from the fitted posterior (with likelihood annealed by
    ``thompson_temperature``), then finds an optimal sequenc wrt those
    parameters via simulated annealing.

    The Bayesian model can be fit either via stochastic variational inference
    (SVI, faster but less accurate) or Markov chain Monte Carlo (MCMC, slower
    but more accurate).

    :param OrderedDict schema: A schema dict.
    :param list constraints: A list of constraints.
    :param list feature_blocks: A list of choice blocks for linear regression.
    :param list gibbs_blocks: A list of choice blocks for Gibbs sampling.
    :param dict experiment: A dict containing all old experiment data.
    :param int design_size: Number of designs to try to return (sometimes
        fewer designs are found).
    :param callable feature_fn: An optional callback to generate additional
        features.
    :param str response_type: Type of response, one of: "real", "unit_interval".
    :param str inference: Inference algorithm, one of: "svi", "mcmc".
    :param int mcmc_num_samples: If ``inference == "mcmc"``, this sets the
        number of posterior samples to draw from MCMC. Should be larger than
        ``design_size``.
    :param int mcmc_warmup_steps: If ``inference == "mcmc", this sets the
        number of warmup steps for MCMC. Should be the same order of magnitude
        as ``mcmc_num_samples``.
    :param int svi_num_steps: If ``inference == "svi"`` this sets the number of
        steps to run stochastic variational inference.
    :param bool svi_reparam: Whether to reparametrize SVI inference.
        This only works when ``thompson_temperature == 1``.
    :param int sa_num_steps: Number of steps to run simulated annealing, at
        each Thompson sample.
    :param bool svi_plot: If ``inference == "svi"`` whether to plot loss curve.
    :param int max_tries: Number of extra Thompson samples to draw in search
        of novel sequences to add to the design.
    :param float thompson_temperature: Likelihood annealing temperature at
        which Thompson samples are drawn. Defaults to 1. You may want to
        increase this if you are have trouble finding novel designs, i.e. if
        this function returns fewer designs than you request.
    :param bool jit_compile: Optional flag to force jit compilation during
        inference. Defaults to safe values for both SVI and MCMC inference.
    :param int log_every: Logging interval for internal algorithms. To disable
        logging, set this to zero.

    :returns: A design as a set of tuples of choices.
    :rtype: set
    """
    if jit_compile is None:
        if inference == "svi":
            jit_compile = False  # default to False to avoid jit errors
        elif inference == "mcmc":
            jit_compile = True  # default to True for speed
        else:
            raise ValueError(f"Unknown inference type: {inference}")

    # Compute extra features.
    extra_features = None
    if feature_fn is not None:
        with torch.no_grad():
            extra_features = feature_fn(experiment["sequences"])
        assert isinstance(extra_features, torch.Tensor)

    # Pass max_batch_id separately as a python int to allow jitting.
    max_batch_id = int(experiment["batch_ids"].max())
    assert thompson_temperature > 0
    bound_model = functools.partial(
        model,
        schema,
        feature_blocks,
        extra_features,
        experiment,
        max_batch_id=max_batch_id,
        response_type=response_type,
        likelihood_temperature=thompson_temperature,
    )
    # Reparametrization can improve variational inference,
    # but isn't compatible with with jit compilation or mcmc.
    if inference == "svi" and svi_reparam:
        bound_model = AutoReparam()(bound_model)
        poutine.block(bound_model)()  # initialize reparam

    # Fit a posterior distribution over parameters given experiment data.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "torch.tensor results are registered as constants",
            torch.jit.TracerWarning,
        )
        if inference == "svi":
            sampler = fit_svi(
                bound_model,
                num_steps=svi_num_steps,
                plot=svi_plot,
                jit_compile=jit_compile,
                log_every=log_every,
            )
        elif inference == "mcmc":
            assert mcmc_num_samples >= design_size
            sampler = fit_mcmc(
                bound_model,
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

        for i in range(design_size + max_tries):
            if log_every:
                print(".", end="", flush=True)
            coefs = poutine.condition(bound_model, sampler())()

            seq = optimize_simulated_annealing(
                schema,
                constraints,
                gibbs_blocks,
                coefs,
                feature_fn=feature_fn,
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
