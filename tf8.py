# type: ignore

import argparse
from collections import OrderedDict

import pyro
import torch
import torch.multiprocessing as mp

from pyroed.criticism import criticize
from pyroed.datasets.data import load_tf_data
from pyroed.oed import thompson_sample

# Specify the design space via SCHEMA, CONSTRAINTS, FEATURES, and GIBBS_BLOCKS.
nucleotides = ["A", "T", "C", "G"]
SCHEMA = OrderedDict((f"N{i}", nucleotides) for i in range(8))

CONSTRAINTS = []

FEATURES = [[name] for name in SCHEMA]
FEATURES.extend([f1 + f2 for f1, f2 in zip(FEATURES, FEATURES[1:])])

GIBBS_BLOCKS = [
    ["N0", "N1", "N2"],
    ["N1", "N2", "N3"],
    ["N2", "N3", "N4"],
    ["N3", "N4", "N5"],
    ["N4", "N5", "N6"],
    ["N5", "N6", "N7"],
]


def main(args):
    pyro.set_rng_seed(args.seed)

    complete_experiment = load_tf_data()

    # Choose an initial batch.
    complete_size = len(complete_experiment["response"])
    complete_ids = torch.randperm(complete_size)
    init_ids = complete_ids[: args.num_initial_sequences]
    test_ids = complete_ids[
        args.num_initial_sequences : args.num_initial_sequences + 50
    ]
    experiment = {k: v[init_ids] for k, v in complete_experiment.items()}
    test_data = {k: v[test_ids] for k, v in complete_experiment.items()}

    criticize(
        SCHEMA,
        CONSTRAINTS,
        FEATURES,
        GIBBS_BLOCKS,
        experiment,
        test_data,
        inference="mcmc" if args.mcmc else "svi",
        mcmc_num_samples=args.mcmc_num_samples,
        mcmc_warmup_steps=args.mcmc_warmup_steps,
        mcmc_num_chains=args.mcmc_num_chains,
        svi_num_steps=args.svi_num_steps,
        log_every=args.log_every,
        jit_compile=args.jit,
    )

    # Perform first active learning step.
    design = thompson_sample(
        SCHEMA,
        CONSTRAINTS,
        FEATURES,
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

    # Simulation parameters.
    parser.add_argument("--num-initial-sequences", default=100, type=int)
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
    parser.add_argument("--svi-num-steps", default=301, type=int)
    parser.add_argument("--sa-num-steps", default=101, type=int)
    parser.add_argument("--jit", default=True, action="store_true")
    parser.add_argument("--nojit", dest="jit", action="store_false")
    parser.add_argument("--seed", default=20210929)
    parser.add_argument("--log-every", default=100, type=int)
    args = parser.parse_args()
    main(args)
