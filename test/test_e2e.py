import os
import tempfile
from collections import OrderedDict
from typing import List, Optional

import pytest
import torch

from pyroed import (
    decode_design,
    encode_design,
    get_next_design,
    start_experiment,
    update_experiment,
)
from pyroed.constraints import AllDifferent
from pyroed.criticism import criticize
from pyroed.typing import Constraints, Schema


def example_feature_fn(sequence):
    sequence = sequence.to(torch.get_default_dtype())
    return torch.stack(
        [
            sequence.sum(-1),
            sequence.max(-1).values,
            sequence.min(-1).values,
            sequence.mean(-1),
            sequence.std(-1),
        ],
        dim=-1,
    )


@pytest.mark.parametrize(
    "response_type, feature_fn",
    [
        ("unit_interval", None),
        ("real", None),
        ("real", example_feature_fn),
    ],
)
@pytest.mark.parametrize(
    "inference, jit_compile",
    [
        ("svi", False),
        pytest.param("svi", True, marks=[pytest.mark.xfail(reason="jit error")]),
        pytest.param("mcmc", False, marks=[pytest.mark.skip(reason="slow")]),
        ("mcmc", True),
    ],
)
def test_end_to_end(inference, jit_compile, response_type, feature_fn):
    # Declare a problem.
    SCHEMA: Schema = OrderedDict()
    SCHEMA["foo"] = ["a", "b", None]
    SCHEMA["bar"] = ["a", "b", "c", None]
    SCHEMA["baz"] = ["a", "b", "c", "d", None]

    CONSTRAINTS: Constraints = [AllDifferent("bar", "baz")]

    FEATURE_BLOCKS = [["foo"], ["bar", "baz"]]

    GIBBS_BLOCKS = [["foo", "bar"], ["bar", "baz"]]

    design_size = 4
    design: List[List[Optional[str]]] = [
        ["a", "b", "c"],
        ["b", "c", "a"],
        ["a", "b", None],
        ["b", "c", None],
    ]

    # Initialize an experiment.
    sequences = encode_design(SCHEMA, design)
    if response_type == "unit_interval":
        responses = torch.rand(design_size)
    elif response_type == "real":
        responses = torch.rand(design_size)
    batch_ids = torch.zeros(design_size, dtype=torch.long)
    experiment = start_experiment(SCHEMA, sequences, responses, batch_ids)

    # Draw new batches.
    config = {
        "response_type": response_type,
        "inference": inference,
        "jit_compile": jit_compile,
        "mcmc_num_samples": 100,
        "mcmc_warmup_steps": 100,
        "svi_num_steps": 100,
        "sa_num_steps": 100,
        "log_every": 10,
    }
    for step in range(2):
        sequences = get_next_design(
            SCHEMA,
            CONSTRAINTS,
            FEATURE_BLOCKS,
            GIBBS_BLOCKS,
            experiment,
            design_size=design_size,
            feature_fn=feature_fn,
            config=config,
        )

        design = decode_design(SCHEMA, sequences)
        actual_sequences = encode_design(SCHEMA, design)
        assert torch.allclose(actual_sequences, sequences)
        responses = torch.rand(design_size)

        if step == 0:
            experiment = update_experiment(SCHEMA, experiment, sequences, responses)
            assert len(experiment["sequences"]) == design_size * (2 + step)
        else:
            test_data = {
                "sequences": sequences,
                "responses": responses,
            }

    # Criticize.
    with tempfile.TemporaryDirectory() as dirname:
        criticize(
            SCHEMA,
            CONSTRAINTS,
            FEATURE_BLOCKS,
            GIBBS_BLOCKS,
            experiment,
            test_data,
            feature_fn=feature_fn,
            filename=os.path.join(dirname, "criticize.pdf"),
        )
