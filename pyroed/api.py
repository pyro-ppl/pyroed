from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

import torch

from .oed import thompson_sample
from .typing import Blocks, Constraints, Schema, validate


def encode_design(
    schema: Schema, design: Iterable[List[Optional[str]]]
) -> torch.Tensor:
    """
    Converts a human readable list of sequences into a tensor.

    :param OrderedDict schema: A schema dict.
    :param list design: A list of list of choices (strings or None).
    :returns: A tensor of encoded sequences.
    :rtype: torch.tensor
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

    :param OrderedDict schema: A schema dict.
    :param torch.Tensor sequences: A tensor of encoded sequences.
    :returns: A list of list of choices (strings or None).
    :rtype: list
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


def start_experiment(
    schema: Schema,
    experiment: Dict[str, torch.Tensor],
    sequences: torch.Tensor,
    responses: torch.Tensor,
    batch_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Creates a cumulative experiment with initial data.

    :param OrderedDict schema: A schema dict.
    :param dict experiment: A dict containing all old experiment data.
    :param torch.Tensor sequences: A tensor of encoded sequences that have been
        measured.
    :param torch.Tensor responses: A tensor of the measured responses of sequences.
    :param torch.Tensor batch_ids: An optional tensor of batch ids.
    :param torch.Tensor
    :returns: A cumulative experiment dict.
    :rtype: dict
    """
    # If unspecified, simply create a single batch id.
    if batch_ids is None:
        batch_ids = sequences.new_zeros(responses.shape)

    # This function is a thin wrapper around dict().
    experiment = {
        "sequences": sequences,
        "response": responses,
        "batch_id": batch_ids,
    }

    # Validate.
    if __debug__:
        validate(schema, experiment=experiment)
    return experiment


def get_next_design(
    schema: Schema,
    constraints: Constraints,
    feature_blocks: Blocks,
    gibbs_blocks: Blocks,
    experiment: Dict[str, torch.Tensor],
    *,
    design_size: int = 10,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Generate a new design given cumulative experimental data.

    :param OrderedDict schema: A schema dict.
    :param list constraints: A list of constraints.
    :param list feature_blocks: A list of choice blocks for linear regression.
    :param list gibbs_blocks: A list of choice blocks for Gibbs sampling.
    :param dict experiment: A dict containing all old experiment data.
    :returns: A tensor of encoded new sequences to measure, i.e. a ``design``.
    :rtype: torch.Tensor
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
    design_set = thompson_sample(
        schema,
        constraints,
        feature_blocks,
        gibbs_blocks,
        experiment,
        design_size=design_size,
        **config,
    )
    design = torch.tensor(sorted(design_set))
    return design


def update_experiment(
    schema: Schema,
    experiment: Dict[str, torch.Tensor],
    new_sequences: torch.Tensor,
    new_responses: torch.Tensor,
    new_batch_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Updates a cumulative experiment by appending new data.

    Note this does not modify its arguments, you must capture the result.

    :param OrderedDict schema: A schema dict.
    :param dict experiment: A dict containing all old experiment data.
    :param torch.Tensor new_sequences: A set of new sequences that have been
        measured. These may simply be the ``design`` returned by
        :func:`get_next_design`, or may be arbitrary new sequences you have
        decided to measure, or old sequences you have measured again, or a
        combination of all three.
    :param torch.Tensor
    :returns: A concatenated experiment.
    :rtype: dict
    """
    # If unspecified, simply create a new single batch id.
    if new_batch_ids is None:
        new_batch_ids = experiment["batch_id"].new_full(
            new_responses.shape, experiment["batch_id"].max().item() + 1
        )

    # Validate.
    if __debug__:
        validate(schema, experiment=experiment)
    assert len(new_responses) == len(new_sequences)
    assert len(new_batch_ids) == len(new_sequences)

    # Concatenate the dictionaries.
    new_experiment = {
        "sequences": new_sequences,
        "response": new_responses,
        "batch_id": new_batch_ids,
    }
    experiment = {k: torch.cat([v, new_experiment[k]]) for k, v in experiment.items()}

    # Validate again.
    if __debug__:
        validate(schema, experiment=experiment)
    return experiment


__all__ = [
    "decode_design",
    "encode_design",
    "get_next_design",
    "start_experiment",
    "update_experiment",
]
