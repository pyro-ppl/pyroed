from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

from .constraints import Constraint
from .oed import thompson_sample
from .typing import Blocks, Constraints, Schema


def encode_design(
    schema: Schema, design: Iterable[List[Optional[str]]]
) -> torch.Tensor:

    rows = []
    for seq in design:
        if len(seq) != len(schema):
            raise ValueError
        row = []
        for value, (name, values) in zip(seq, schema.items()):
            try:
                row.append(values.index(value))
            except ValueError:
                raise ValueError(
                    f"Value {repr(value)} not found in schema[{repr(name)}]"
                )
        rows.append(row)
    return torch.tensor(rows)


def decode_design(schema: Schema, sequences: torch.Tensor) -> List[List[Optional[str]]]:
    """
    Converts an tensor representation of a design into a readable list of designs.
    """
    raise NotImplementedError("TODO")


def get_next_design(
    *,
    schema: Schema,
    constraints: Constraints,
    feature_blocks: Blocks,
    gibbs_blocks: Blocks,
    experiment: Dict[str, torch.Tensor],
    design_size: int = 10,
    config: Optional[dict] = None,
) -> Set[Tuple[int, ...]]:
    if __debug__:
        validate(schema, constraints, feature_blocks, gibbs_blocks, experiment)

    if config is None:
        config = {}
    return thompson_sample(
        schema,
        constraints,
        feature_blocks,
        gibbs_blocks,
        experiment,
        design_size=design_size,
        **config,
    )


def validate(
    schema: Schema,
    constraints: Constraints,
    feature_blocks: Blocks,
    gibbs_blocks: Blocks,
    experiment: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    # Validate schema.
    assert isinstance(schema, OrderedDict)
    for name, values in schema.items():
        assert isinstance(name, str)
        assert isinstance(values, list)
        assert values
        for value in values:
            assert value is None or isinstance(value, str)

    # Validate constraints.
    assert isinstance(constraints, list)
    for constraint in constraints:
        assert isinstance(constraint, Constraint)

    # Validate feature_blocks.
    assert isinstance(feature_blocks, list)
    for block in feature_blocks:
        assert isinstance(block, list)
        for col in block:
            assert col in schema
    assert len({tuple(f) for f in feature_blocks}) == len(
        feature_blocks
    ), "duplicate feature_blocks"

    # Validate gibbs_blocks.
    assert isinstance(gibbs_blocks, list)
    for block in gibbs_blocks:
        assert isinstance(block, list)
        for col in block:
            assert col in schema
    assert len({tuple(f) for f in gibbs_blocks}) == len(
        gibbs_blocks
    ), "duplicate gibbs_blocks"

    # Validate experiment.
    if experiment is not None:
        assert isinstance(experiment, dict)
        assert set(experiment.keys()) == {"sequences", "batch_id", "response"}
        sequences = experiment["sequences"]
        batch_id = experiment["batch_id"]
        response = experiment["response"]
        assert isinstance(sequences, torch.Tensor)
        assert sequences.dtype == torch.long
        assert sequences.dim == 2
        assert sequences.shape[-1] == len(schema)
        assert len(sequences) == len(batch_id)
        assert len(sequences) == len(response)
        assert isinstance(batch_id, torch.Tensor)
        assert batch_id.dtype == torch.long
        assert batch_id.dim() == 1
        assert isinstance(response, torch.Tensor)
        assert response.dtype in (torch.float, torch.double)
        assert response.dim() == 1


__all__ = [
    "decode_design",
    "encode_design",
    "get_next_design",
    "validate",
]
