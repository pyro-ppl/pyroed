import argparse
from collections import OrderedDict

import pyro
import pyro.poutine as poutine
import torch

from pyroed.constraints import AllDifferent, Iff, IfThen, TakesValue
from pyroed.models import model
from pyroed.oed import thompson_sample
from pyroed.typing import Schema

SCHEMA: Schema = OrderedDict(
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
FEATURES = [[name] for name in SCHEMA]
FEATURES.append(["Protein 1", "Protein 2"])  # TODO(liz) add a real interaction
GIBBS_BLOCKS = [
    ["Protein 1", "2A-1"],
    ["Signalling Pep", "EP", "Linker"],
    ["2A-1", "2A-2", "2A-3"],
    ["Protein 2", "Internal", "2A-2"],
]


@torch.no_grad()
def generate_fake_data(N_per_B=10, B=2):
    pyro.set_rng_seed()
    N = N_per_B * B
    experiment = {}
    experiment["batch_id"] = (torch.arange(N) // N_per_B,)
    experiment["sequences"] = torch.stack(
        [torch.randint(0, len(choices), (N,)) for choices in SCHEMA.values()], dim=-1
    )
    trace = poutine.trace(model).get_trace(SCHEMA, FEATURES, experiment)
    truth = truth = {
        name: site["value"].detach()
        for name, site in trace.nodes.items()
        if site["type"] == "sample" and not site["is_observed"]
        if type(site["fn"]).__name__ != "_Subsample"
        if name != "batch_response"  # shape varies in time
    }
    experiment["response"] = trace.nodes["response"]["value"].detach()
    return truth, experiment


def main(args):
    truth, experiment = generate_fake_data(args)
    thompson_sample(
        SCHEMA,
        CONSTRAINTS,
        FEATURES,
        GIBBS_BLOCKS,
        experiment,
        num_svi_steps=args.num_svi_steps,
        num_sa_steps=args.num_sa_steps,
        max_tries=args.max_tries,
        log_every=args.log_every,
        thompson_temperature=args.thompson_temperature,
    )
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Design sequences")
    parser.add_argument("--sequences--per-batch", default=10, type=int)
    parser.add_argument("--simulate-batches", default=20)
    parser.add_argument("--seed", default=20210929)
    parser.add_argument("--num-svi-steps", default=201, type=int)
    parser.add_argument("--num-sa-steps", default=201, type=int)
    parser.add_argument("--max-tries", default=1000, type=int)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--thompson-temperature", default=4.0, type=float)
    args = parser.parse_args()
    main(args)
