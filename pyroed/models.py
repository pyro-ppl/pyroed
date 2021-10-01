from collections import OrderedDict
from typing import Dict

import pyro
import pyro.distributions as dist
import torch

from .typing import Coefs, Features, Schema


def linear_response(schema: Schema, coefs: Coefs, sequence: torch.Tensor):
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
    features: Features,
    experiment: Dict[str, torch.Tensor],  # sequences, batch_id, response
    *,
    quantization_bins=100,
):
    P = len(schema)
    assert experiment["sequences"].dtype == torch.int64
    assert experiment["sequences"].dim() == 2
    assert experiment["sequences"].shape[-1] == P
    N = len(experiment["sequences"])
    if experiment["response"] is not None:
        assert torch.is_floating_point(experiment["response"])
        assert experiment["response"].shape == (N,)
        assert 0 <= experiment["response"].min()
        assert experiment["response"].max() <= 1
    assert experiment["batch_id"].dtype == torch.int64
    assert experiment["batch_id"].shape == (N,)
    B = 1 + int(experiment["batch_id"].max())
    name_to_int = {name: i for i, name in enumerate(schema)}

    # Hierarchically sample linear coefficients.
    coef_scale_loc = pyro.sample("coef_scale_loc", dist.Normal(-2, 1))
    coef_scale_scale = pyro.sample("coef_scale_scale", dist.LogNormal(0, 1))
    coefs: Coefs = {}
    for names in features:
        shape = tuple(len(schema[name]) for name in names)
        ps = tuple(name_to_int[name] for name in names)
        suffix = "_".join(map(str, ps))
        # Within-component variance of coefficients.
        coef_scale = pyro.sample(
            f"coef_scale_{suffix}",
            dist.LogNormal(coef_scale_loc, coef_scale_scale),
        )
        # Linear coefficients. Note this overparametrizes; there are only
        # len(choices) - 1 degrees of freedom and 1 nuisance dim.
        coefs[tuple(names)] = pyro.sample(
            f"coef_{suffix}",
            dist.Normal(torch.zeros(shape), coef_scale).to_event(len(shape)),
        )

    # Compute the linear response function.
    response_loc = linear_response(schema, coefs, experiment["sequences"])

    # Observe a noisy response.
    # This could be changed to counts or whatever.
    across_batch_scale = pyro.sample("across_batch_scale", dist.LogNormal(0, 1))
    within_batch_scale = pyro.sample("within_batch_scale", dist.LogNormal(0, 1))
    with pyro.plate("batch", B):
        batch_response = pyro.sample(
            "batch_response", dist.Normal(0, across_batch_scale)
        )
        assert batch_response.shape == (B,)
    with pyro.plate("data", N):
        logits = pyro.sample(
            "logits",
            dist.Normal(
                response_loc + batch_response[experiment["batch_id"]],
                within_batch_scale,
            ),
        )

        # Quantize the observation to avoid numerical artifacts near 0 and 1.
        quantized_obs = None
        if experiment["response"] is not None:  # during inference
            quantized_obs = (experiment["response"] * quantization_bins).round()
        quantized_obs = pyro.sample(
            "quantized_response",
            dist.Binomial(quantization_bins, logits=logits),
            obs=quantized_obs,
        )
        if experiment["response"] is None:  # during simulation
            pyro.deterministic("response", quantized_obs / quantization_bins)

    return coefs
