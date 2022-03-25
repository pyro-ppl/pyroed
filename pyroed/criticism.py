import functools
import warnings
from typing import Callable, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyro import poutine

from .inference import fit_mcmc, fit_svi
from .models import linear_response, model
from .typing import Blocks, Constraints, Schema

matplotlib.use("Agg")


@torch.no_grad()
def criticize(
    schema: Schema,
    constraints: Constraints,
    feature_blocks: Blocks,
    gibbs_blocks: Blocks,
    experiment: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    *,
    feature_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    response_type: str = "unit_interval",
    inference: str = "svi",
    num_posterior_samples: int = 11,
    mcmc_num_samples: int = 500,
    mcmc_warmup_steps: int = 500,
    mcmc_num_chains: int = 1,
    svi_num_steps: int = 201,
    jit_compile: bool = False,
    log_every: int = 100,
    filename: str = "criticize.pdf",
) -> None:
    """
    Plots observed versus predicted responses on a held out test set.

    :param OrderedDict schema: A schema dict.
    :param list constraints: A list of constraints.
    :param list feature_blocks: A list of choice blocks for linear regression.
    :param list gibbs_blocks: A list of choice blocks for Gibbs sampling.
    :param dict experiment: A dict containing all old experiment data.
    :param dict test_data: A dict containing held out test data.
    """
    # Compute extra features.
    extra_features = None
    if feature_fn is not None:
        extra_features = feature_fn(experiment["sequences"])

    bound_model = functools.partial(
        model,
        schema,
        feature_blocks,
        extra_features,
        experiment,
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
                bound_model,
                num_steps=svi_num_steps,
                jit_compile=jit_compile,
                plot=False,
            )
        elif inference == "mcmc":
            sampler = fit_mcmc(
                bound_model,
                num_samples=mcmc_num_samples,
                warmup_steps=mcmc_warmup_steps,
                num_chains=mcmc_num_chains,
                jit_compile=jit_compile,
            )
        else:
            raise ValueError(f"Unknown inference type: {inference}")

        test_responses = test_data["responses"]
        test_sequences = test_data["sequences"]
        sort_idx = test_responses.sort(0).indices
        test_responses = test_responses[sort_idx]
        test_sequences = test_sequences[sort_idx]
        if feature_fn is not None:
            extra_features = feature_fn(test_sequences)

        predictions = []
        for _ in range(num_posterior_samples):
            coefs = poutine.condition(bound_model, sampler())()
            test_prediction = linear_response(
                schema, coefs, test_sequences, extra_features
            ).sigmoid()
            predictions.append(test_prediction)

        test_predictions = torch.stack(predictions).detach().cpu().numpy()
        mean_predictions = test_predictions.mean(0)
        std_predictions = test_predictions.std(0)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.plot(
            np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            color="k",
            linestyle="dotted",
        )

        plt.errorbar(
            test_responses,
            mean_predictions,
            yerr=std_predictions,
            marker="o",
            linestyle="None",
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ax.set_xlabel("Observed response")
        ax.set_ylabel("Predictedresponse")

        fig.tight_layout()
        plt.savefig(filename)
        print(f"Saved {filename}")
