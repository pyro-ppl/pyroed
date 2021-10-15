# type: ignore

import argparse
import warnings
from collections import OrderedDict

import pandas as pd
import pyro
import torch
import torch.multiprocessing as mp

from pyroed.constraints import AllDifferent, Iff, IfThen, TakesValue
from pyroed.datasets.data import get_tf_data

# Specify the design space via SCHEMA, CONSTRAINTS, FEATURES, and GIBBS_BLOCKS.
SCHEMA = OrderedDict()
SCHEMA["Protein 1"] = ["Prot1", "Prot2", None]
SCHEMA["Protein 2"] = ["Prot3", "HLA1", "HLA2", "HLA3", "HLA4", None]
SCHEMA["Signalling Pep"] = ["Sig1", "Sig2", None]
SCHEMA["EP"] = [f"Ep{i}" for i in range(1, 10 + 1)] + [None]
SCHEMA["Linker"] = ["Link1", None]
SCHEMA["Internal"] = ["Int1", "Int2", "Int3", "Int3", None]
SCHEMA["2A-1"] = ["twoa1", "twoa2", None]
SCHEMA["2A-2"] = ["twoa3", "twoa4", None]
SCHEMA["2A-3"] = [f"twoa{i}" for i in range(1, 7 + 1)]

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


def load_experiment(filename, schema):
    df = pd.read_csv(filename, sep="\t")

    # Load response.
    col = "Amount Expression Output 1"
    df = df[~df[col].isna()]  # Filter to rows where response was observed.
    N = len(df[col])
    response = torch.zeros(N)
    response[:] = torch.tensor([float(cell.strip("%")) / 100 for cell in df[col]])

    # Load sequences.
    sequences = torch.zeros(N, len(SCHEMA), dtype=torch.long)
    for i, (name, values) in enumerate(SCHEMA.items()):
        sequences[:, i] = torch.tensor(
            [values.index(v if isinstance(v, str) else None) for v in df[name]]
        )

    # Optionally load batch id.
    col = "Batch ID"
    batch_id = torch.zeros(N, dtype=torch.long)
    if col in df:
        batch_id[:] = df[col].to_numpy()
    else:
        warnings.warn(f"Found no '{col}' column, assuming a single batch")

    return {
        "sequences": sequences,
        "batch_id": batch_id,
        "response": response,
    }


def main(args):
    pyro.set_rng_seed(args.seed)

    x, y = get_tf_data()
    print("xy", x.shape, y.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Design sequences")

    # Simulation parameters.
    parser.add_argument("--sequences-per-batch", default=10, type=int)
    parser.add_argument("--simulate-batches", default=20)

    # Algorithm parameters.
    parser.add_argument("--max-tries", default=1000, type=int)
    parser.add_argument("--thompson-temperature", default=4.0, type=float)
    parser.add_argument("--mcmc", default=False, action="store_true")
    parser.add_argument("--svi", dest="mcmc", action="store_false")
    parser.add_argument("--mcmc-num-samples", default=500, type=int)
    parser.add_argument("--mcmc-warmup-steps", default=500, type=int)
    parser.add_argument("--mcmc-num-chains", default=min(4, mp.cpu_count()), type=int)
    parser.add_argument("--svi-num-steps", default=201, type=int)
    parser.add_argument("--sa-num-steps", default=201, type=int)
    parser.add_argument("--jit", default=True, action="store_true")
    parser.add_argument("--nojit", dest="jit", action="store_false")
    parser.add_argument("--seed", default=20210929)
    parser.add_argument("--log-every", default=100, type=int)
    args = parser.parse_args()
    main(args)
