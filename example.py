from collections import OrderedDict
from typing import Dict

import pyro
import pyro.distributions as dist
import torch

from pyroed.constraints import AllDifferent, Iff, IfThen, TakesValue

SCHEMA = OrderedDict(
    [
        ("Protein 1", ["Prot1", "Prot2", None]),
        ("Protein 2", ["Prot3", "HLA1", "HLA2", "HLA3", "HLA4"]),
        ("Signalling Pep", ["Sig1", "Sig2", None]),
        ("EP", [f"EP{i}" for i in range(1, 10 + 1)]),
        ("Linker", ["Link1", None]),
        ("Internal", ["Int1", "Int2", "Int3", "Int3", None]),
        ("2A-1", ["twoa1", "twoa2", None]),
        ("2A-2", ["twoa3", "twoa4", None]),
        ("2A-3", [f"twoa{i}" for i in range(1, 7 + 1)]),
    ]
)
CONSTRAINTS = [
    AllDifferent("2A-1", "2A-2", "2A-3"),
    Iff(TakesValue("Protein 1", None), TakesValue("2A-1", None)),
    Iff(TakesValue("Signalling Pep", None), TakesValue("EP", None)),
    Iff(TakesValue("EP", None), TakesValue("Linker", None)),
    IfThen(TakesValue("Protein 2", None), TakesValue("Internal", None)),
    Iff(TakesValue("Protein 2", "Prot3"), TakesValue("2A-2", None)),
]
GIBBS_BLOCKS = [
    ["Protein 1", "2A-1"],
    ["Signalling Pep", "EP", "Linker"],
    ["2A-1", "2A-2", "2A-3"],
    ["Protein 2", "Internal", "2A-2"],
]


def response_fn(coefs: Dict[str, torch.Tensor], sequence: torch.Tensor):
    # Separate individual choices.
    assert sequence.dtype == torch.long
    assert sequence.size(-1) == len(SCHEMA)
    choices = dict(zip(SCHEMA, sequence.unbind(-1)))

    # Coefficients.
    response = 0.0
    for name, coef in coefs.items():
        assert coef.dim() == 1  # a vector of coefficients
        response = response + coef[choices[name]]
    return response


def model(
    experiment_sequences: torch.Tensor,  # covariates
    experiment_response: torch.Tensor,  # response
    experiment_batch_id: torch.Tensor,  # batch effects
    *,
    quantization_bins=100,
):
    P = len(SCHEMA)
    assert experiment_sequences.dtype == torch.int64
    assert experiment_sequences.dim() == 2
    assert experiment_sequences.shape[-1] == P
    N = len(experiment_sequences)
    if experiment_response is not None:
        assert torch.is_floating_point(experiment_response)
        assert experiment_response.shape == (N,)
        assert 0 <= experiment_response.min()
        assert experiment_response.max() <= 1
    assert experiment_batch_id.dtype == torch.int64
    assert experiment_batch_id.shape == (N,)
    B = 1 + int(experiment_batch_id.max())

    # How do componentwise coefficients vary among components.
    coef_scale_loc = pyro.sample("coef_scale_loc", dist.Normal(-2, 1))
    coef_scale_scale = pyro.sample("coef_scale_scale", dist.LogNormal(0, 1))

    # Component-wise coefficients.
    coefs = {}
    for p, (name, choices) in enumerate(SCHEMA.items()):  # morally a plate
        # How much do coefficients vary within a component.
        coef_scale = pyro.sample(
            f"coef_scale_{p}",
            dist.LogNormal(coef_scale_loc, coef_scale_scale),
        )
        # The linear coefficients for component p.
        # Note this overparametrizes; there should really be
        # only len(choices) - 1 degrees of freedom.
        coefs[name] = pyro.sample(
            f"coef_{p}",
            dist.Normal(torch.zeros(len(choices)), coef_scale).to_event(1),
        )

    # TODO add effects of interacting pairs.

    response_loc = response_fn(coefs, experiment_sequences)

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
            "response",
            dist.Normal(
                response_loc + batch_response[experiment_batch_id], within_batch_scale
            ),
        )
        quantized_obs = None
        if experiment_response is not None:  # during inference
            quantized_obs = (experiment_response * quantization_bins).round()
        quantized_obs = pyro.sample(
            "quantized_response",
            dist.Binomial(quantization_bins, logits=logits),
            obs=quantized_obs,
        )
        if experiment_response is None:  # during simulation
            pyro.deterministic("response", quantized_obs / quantization_bins)

    return coefs


def generate_fake_data(N_per_B=10, B=2):
    N = N_per_B * B
    experiment_batch_id = (torch.arange(N) // N_per_B,)
    experiment_sequences = torch.stack(
        [torch.randint(0, len(choices), (N,)) for choices in SCHEMA.values()], dim=-1
    )
    experiment_response = torch.rand(N)  # FIXME run the model
    return dict(
        experiment_batch_id=experiment_batch_id,
        experiment_sequences=experiment_sequences,
        experiment_response=experiment_response,
    )


def main():
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
