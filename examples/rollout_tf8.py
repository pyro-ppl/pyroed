# type: ignore

import argparse
import pickle
import time
from collections import OrderedDict

import pyro
import torch

from pyroed.datasets.data import load_tf_data
from pyroed.oed import thompson_sample

SCHEMA = OrderedDict()
for n in range(8):
    SCHEMA[f"Nucleotide{n}"] = ["A", "C", "G", "T"]

CONSTRAINTS = []  # No constraints.

singletons = [[name] for name in SCHEMA]
pairs = [list(ns) for ns in zip(SCHEMA, list(SCHEMA)[1:])]
triples = [list(ns) for ns in zip(SCHEMA, list(SCHEMA)[1:], list(SCHEMA)[2:])]

SINGLETON_BLOCKS = singletons
PAIRWISE_BLOCKS = singletons + pairs
GIBBS_BLOCKS = triples


def update_experiment(experiment: dict, design: set, data: dict) -> dict:
    ids = list(map(data["seq_to_id"].__getitem__, sorted(design)))
    new_data = {
        "sequences": data["sequences"][ids],
        "responses": data["responses"][ids],
        "batch_ids": torch.zeros(len(ids)).long(),
    }
    experiment = {k: torch.cat([v, new_data[k]]) for k, v in experiment.items()}
    return experiment


def make_design(
    experiment: dict,
    design_size: int,
    thompson_temperature: float,
    feature_blocks: list,
) -> set:
    return thompson_sample(
        SCHEMA,
        CONSTRAINTS,
        feature_blocks,
        GIBBS_BLOCKS,
        experiment,
        design_size=design_size,
        thompson_temperature=thompson_temperature,
        inference="svi",
        svi_num_steps=1000,
        sa_num_steps=400,
        log_every=0,
    )


def main(args):
    pyro.set_rng_seed(args.seed)

    data = load_tf_data()
    ids = torch.randperm(len(data["responses"]))[: args.num_initial_sequences]
    experiment = {k: v[ids] for k, v in data.items()}
    data["seq_to_id"] = {
        tuple(row): i for i, row in enumerate(data["sequences"].tolist())
    }

    print(
        f"[Initial batch] Best response thus far: {experiment['response'].max():0.6g}"
    )
    experiments = [experiment]
    t0 = time.time()

    for batch in range(args.num_batches):
        design = make_design(
            experiments[-1],
            args.num_sequences_per_batch,
            args.thompson_temperature,
            SINGLETON_BLOCKS if args.features == "singleton" else PAIRWISE_BLOCKS,
        )
        experiments.append(update_experiment(experiments[-1], design, data))
        print(
            "[Batch #{}] Best response thus far: {:0.6g}".format(
                batch + 1, experiments[-1]["responses"].max().item()
            )
        )

    print(
        "Best response from all batches: {:0.6g}".format(
            experiments[-1]["responses"].max().item()
        )
    )
    print("Elapsed time: {:.4f}".format(time.time() - t0))

    response_curve = [e["responses"].max().item() for e in experiments]

    f = "results/results.{}.s{}.temp{}.nb{}.nspb{}.nis{}.pkl"
    f = f.format(
        args.features,
        args.seed,
        int(args.thompson_temperature),
        args.num_batches,
        args.num_sequences_per_batch,
        args.num_initial_sequences,
    )
    pickle.dump(response_curve, open(f, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Design sequences")

    parser.add_argument("--num-initial-sequences", default=30, type=int)
    parser.add_argument("--num-sequences-per-batch", default=10, type=int)
    parser.add_argument("--num-batches", default=7)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--thompson-temperature", default=4.0, type=float)
    parser.add_argument(
        "--features", type=str, default="singleton", choices=["singleton", "pairwise"]
    )

    args = parser.parse_args()

    main(args)
