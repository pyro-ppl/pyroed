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
from pyroed.typing import Constraints, Schema


@pytest.mark.parametrize("response_type", ["unit_interval", "real"])
@pytest.mark.parametrize(
    "inference, jit_compile",
    [
        ("svi", False),
        pytest.param("svi", True, marks=[pytest.mark.xfail(reason="jit error")]),
        pytest.param("mcmc", False, marks=[pytest.mark.skip(reason="slow")]),
        ("mcmc", True),
    ],
)
def test_end_to_end(inference, jit_compile, response_type):
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

    sequences = encode_design(SCHEMA, design)
    responses = torch.rand(design_size)
    batch_ids = torch.zeros(design_size, dtype=torch.long)
    experiment = start_experiment(SCHEMA, sequences, responses, batch_ids)

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
            config=config,
        )

        design = decode_design(SCHEMA, sequences)
        actual_sequences = encode_design(SCHEMA, design)
        assert torch.allclose(actual_sequences, sequences)
        responses = torch.rand(design_size)
        experiment = update_experiment(SCHEMA, experiment, sequences, responses)
        assert len(experiment["sequences"]) == design_size * (2 + step)
