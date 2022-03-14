# type: ignore

import argparse
import warnings
from collections import OrderedDict

import pandas as pd
import pyro
import torch
import torch.multiprocessing as mp

from pyroed.constraints import AllDifferent, Iff, IfThen, TakesValue
from pyroed.oed import thompson_sample
from pyroed.testing import generate_fake_data

# Specify the design space via SCHEMA, CONSTRAINTS, FEATURE_BLOCKS, and GIBBS_BLOCKS.
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

FEATURE_BLOCKS = [[name] for name in SCHEMA]
FEATURE_BLOCKS.append(["Protein 1", "Protein 2"])  # TODO(liz) add a real interaction

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

    if args.tsv_file_in:
        print(f"Loading data from {args.tsv_file_in}")
        experiment = load_experiment(args.tsv_file_in, SCHEMA)
    else:
        print("Generating fake data")
        truth, experiment = generate_fake_data(
            SCHEMA, FEATURE_BLOCKS, args.sequences_per_batch, args.simulate_batches
        )

    design = thompson_sample(
        SCHEMA,
        CONSTRAINTS,
        FEATURE_BLOCKS,
        GIBBS_BLOCKS,
        experiment,
        inference="mcmc" if args.mcmc else "svi",
        mcmc_num_samples=args.mcmc_num_samples,
        mcmc_warmup_steps=args.mcmc_warmup_steps,
        mcmc_num_chains=args.mcmc_num_chains,
        svi_num_steps=args.svi_num_steps,
        sa_num_steps=args.sa_num_steps,
        max_tries=args.max_tries,
        thompson_temperature=args.thompson_temperature,
        log_every=args.log_every,
        jit_compile=args.jit,
    )
    print("Design:")
    for row in sorted(design):
        cells = [values[i] for values, i in zip(SCHEMA.values(), row)]
        print("\t".join("-" if c is None else c for c in cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Design sequences")

    # Data files.
    parser.add_argument("--tsv-file-in")

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
    parser.add_argument("--jit", default=False, action="store_true")
    parser.add_argument("--nojit", dest="jit", action="store_false")
    parser.add_argument("--seed", default=20210929)
    parser.add_argument("--log-every", default=100, type=int)
    args = parser.parse_args()
    main(args)
