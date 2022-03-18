from collections import OrderedDict
from typing import Dict, Optional

import pyro
import pyro.distributions as dist
import torch

from .typing import Blocks, Coefs, Schema, validate


def linear_response(
    schema: Schema,
    coefs: Coefs,
    sequence: torch.Tensor,
) -> torch.Tensor:
    """
    Linear response function.

    :param OrderedDict schema: A schema dict.
    :param dict coefs: A dictionary mapping feature tuples to coefficient
        tensors.
    :param torch.Tensor sequence: A tensor representing a sequence.
    :returns: The response.
    :rtype: torch.Tensor
    """
    if not torch._C._get_tracing_state():
        assert isinstance(schema, OrderedDict)
        assert isinstance(coefs, dict)
        assert sequence.dtype == torch.long
        assert sequence.size(-1) == len(schema)
    choices = dict(zip(schema, sequence.unbind(-1)))

    result = torch.tensor(0.0)
    for key, coef in coefs.items():
        assert isinstance(key, tuple)
        assert coef.dim() == len(key)
        index = tuple(choices[name] for name in key)
        result = result + coef[index]

    return result


def model(
    schema: Schema,
    feature_blocks: Blocks,
    experiment: Dict[str, torch.Tensor],  # sequences, batch_id, optional(response)
    *,
    max_batch_id: Optional[int] = None,
    quantization_bins=100,
):
    """
    A `Pyro <https://pyro.ai>`_ model for Bayesian linear regression.

    :param OrderedDict schema: A schema dict.
    :param list feature_blocks: A list of choice blocks for linear regression.
    :param dict experiment: A dict containing all old experiment data.
    :param int quantization_bins: Number of bins in which to quantize the
        response likelihood.
    :returns: A dictionary mapping feature tuples to coefficient tensors.
    :rtype: dict
    """
    if max_batch_id is None:
        max_batch_id = int(experiment["batch_ids"].max())
    N = experiment["sequences"].size(0)
    B = 1 + max_batch_id
    if __debug__ and not torch._C._get_tracing_state():
        validate(schema, experiment=experiment)
    name_to_int = {name: i for i, name in enumerate(schema)}

    # Hierarchically sample linear coefficients.
    coef_scale_loc = pyro.sample("coef_scale_loc", dist.Normal(-2, 1))
    coef_scale_scale = pyro.sample("coef_scale_scale", dist.LogNormal(0, 1))
    coefs: Coefs = {}
    for block in feature_blocks:
        shape = tuple(len(schema[name]) for name in block)
        ps = tuple(name_to_int[name] for name in block)
        suffix = "_".join(map(str, ps))
        # Within-component variance of coefficients.
        coef_scale = pyro.sample(
            f"coef_scale_{suffix}",
            dist.LogNormal(coef_scale_loc, coef_scale_scale),
        )
        # Linear coefficients. Note this overparametrizes; there are only
        # len(choices) - 1 degrees of freedom and 1 nuisance dim.
        coefs[tuple(block)] = pyro.sample(
            f"coef_{suffix}",
            dist.Normal(torch.zeros(shape), coef_scale).to_event(len(shape)),
        )

    # Compute the linear response function.
    response_loc = linear_response(schema, coefs, experiment["sequences"])

    # Observe a noisy response.
    # This likelihood could be generalized to counts or whatever.
    within_batch_scale = pyro.sample("within_batch_scale", dist.LogNormal(0, 1))
    if B == 1:
        batch_response = torch.zeros(B)
    else:
        # Model batch effects.
        across_batch_scale = pyro.sample("across_batch_scale", dist.LogNormal(0, 1))
        with pyro.plate("batch", B):
            batch_response = pyro.sample(
                "batch_response", dist.Normal(0, across_batch_scale)
            )
            if not torch._C._get_tracing_state():
                assert batch_response.shape == (B,)
    with pyro.plate("data", N):
        logits = pyro.sample(
            "logits",
            dist.Normal(
                response_loc + batch_response[experiment["batch_ids"]],
                within_batch_scale,
            ),
        )

        # Quantize the observation to avoid numerical artifacts near 0 and 1.
        quantized_obs = None
        response = experiment.get("responses")
        if response is not None:  # during inference
            quantized_obs = (response * quantization_bins).round()
        quantized_obs = pyro.sample(
            "quantized_response",
            dist.Binomial(quantization_bins, logits=logits),
            obs=quantized_obs,
        )
        if response is None:  # during simulation
            pyro.deterministic("responses", quantized_obs / quantization_bins)

    return coefs
