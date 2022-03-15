import warnings
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyro import poutine

from .inference import fit_mcmc, fit_svi
from .models import linear_response, model
from .typing import Blocks, Constraints, Schema

matplotlib.use("Agg")


def criticize(
    schema: Schema,
    constraints: Constraints,
    feature_blocks: Blocks,
    gibbs_blocks: Blocks,
    experiment: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    *,
    inference="svi",
    num_posterior_samples=11,
    mcmc_num_samples=500,
    mcmc_warmup_steps=500,
    mcmc_num_chains=1,
    svi_num_steps=201,
    jit_compile=True,
    log_every=100,
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

    def tf8_model():
        return model(schema, feature_blocks, experiment)

    # Fit a posterior distribution over parameters given experiment data.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "torch.tensor results are registered as constants",
            torch.jit.TracerWarning,
        )
        if inference == "svi":
            sampler = fit_svi(
                tf8_model,
                num_steps=svi_num_steps,
                jit_compile=jit_compile,
                plot=False,
            )
        elif inference == "mcmc":
            sampler = fit_mcmc(
                tf8_model,
                num_samples=mcmc_num_samples,
                warmup_steps=mcmc_warmup_steps,
                num_chains=mcmc_num_chains,
                jit_compile=jit_compile,
            )
        else:
            raise ValueError(f"Unknown inference type: {inference}")

        test_responses = test_data["responses"].numpy()
        test_sequences = test_data["sequences"].numpy()

        sort_idx = np.argsort(test_responses)
        test_responses = test_responses[sort_idx]
        test_sequences = test_sequences[sort_idx]

        predictions = []
        for _ in range(num_posterior_samples):
            with poutine.condition(data=sampler()):
                coefs = model(schema, feature_blocks, experiment)
                test_prediction = linear_response(
                    schema, coefs, test_sequences
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
            mean_predictions.numpy(),
            yerr=std_predictions.numpy(),
            marker="o",
            linestyle="None",
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ax.set_xlabel("Observed response")
        ax.set_ylabel("Predictedresponse")

        fig.tight_layout()
        plt.savefig("criticize.pdf")
        print("Saving criticize.pdf")
