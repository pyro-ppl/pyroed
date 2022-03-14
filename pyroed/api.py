from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

from .oed import thompson_sample
from .typing import Blocks, Constraints, Schema, validate


def encode_design(
    schema: Schema, design: Iterable[List[Optional[str]]]
) -> torch.Tensor:
    """
    Converts a human readable list of design into a tensor.
    """
    # Validate inputs.
    design = list(design)
    assert len(design) > 0
    assert isinstance(schema, OrderedDict)
    if __debug__:
        for seq in design:
            assert len(seq) == len(schema)
            for value, (name, values) in zip(seq, schema.items()):
                if value not in values:
                    raise ValueError(
                        f"Value {repr(value)} not found in schema[{repr(name)}]"
                    )

    # Convert python list -> tensor.
    rows = [
        [values.index(value) for value, values in zip(seq, schema.values())]
        for seq in design
    ]
    return torch.tensor(rows, dtype=torch.long)


def decode_design(schema: Schema, sequences: torch.Tensor) -> List[List[Optional[str]]]:
    """
    Converts an tensor representation of a design into a readable list of designs.
    """
    # Validate.
    assert isinstance(schema, OrderedDict)
    assert isinstance(sequences, torch.Tensor)
    assert sequences.dtype == torch.long
    assert sequences.dim() == 2

    # Convert tensor -> python list.
    rows = [
        [values[i] for i, values in zip(seq, schema.values())]
        for seq in sequences.tolist()
    ]
    return rows


def get_next_design(
    schema: Schema,
    constraints: Constraints,
    feature_blocks: Blocks,
    gibbs_blocks: Blocks,
    experiment: Dict[str, torch.Tensor],
    *,
    design_size: int = 10,
    config: Optional[dict] = None,
) -> Set[Tuple[int, ...]]:
    """
    Generate a new design given cumulative experimental data.

    :param OrderedDict schema: A schema dict.
    """
    if config is None:
        config = {}

    # Validate inputs.
    assert isinstance(design_size, int)
    assert design_size > 0
    assert isinstance(config, dict)
    if __debug__:
        validate(
            schema,
            constraints=constraints,
            feature_blocks=feature_blocks,
            gibbs_blocks=gibbs_blocks,
            experiment=experiment,
        )

    # Perform OED via Thompson sampling.
    design = thompson_sample(
        schema,
        constraints,
        feature_blocks,
        gibbs_blocks,
        experiment,
        design_size=design_size,
        **config,
    )
    return design


__all__ = [
    "decode_design",
    "encode_design",
    "get_next_design",
]
