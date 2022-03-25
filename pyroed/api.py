from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional

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
    sequences: torch.Tensor,
    responses: torch.Tensor,
    batch_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Creates a cumulative experiment with initial data.

    :param OrderedDict schema: A schema dict.
    :param torch.Tensor sequences: A tensor of encoded sequences that have been
        measured.
    :param torch.Tensor responses: A tensor of the measured responses of sequences.
    :param torch.Tensor batch_ids: An optional tensor of batch ids.
    :returns: A cumulative experiment dict.
    :rtype: dict
    """
    # If unspecified, simply create a single batch id.
    if batch_ids is None:
        batch_ids = sequences.new_zeros(responses.shape)

    # This function is a thin wrapper around dict().
    experiment = {
        "sequences": sequences,
        "responses": responses,
        "batch_ids": batch_ids,
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
    feature_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Generate a new design given cumulative experimental data.

    :param OrderedDict schema: A schema dict.
    :param list constraints: A list of constraints.
    :param list feature_blocks: A list of choice blocks for linear regression.
    :param list gibbs_blocks: A list of choice blocks for Gibbs sampling.
    :param dict experiment: A dict containing all old experiment data.
    :param int design_size: Number of designs to try to return (sometimes
        fewer designs are found).
    :param callable feature_fn: An optional callback to generate additional
        features. If provided, this function should input a batch of sequences
        (say of shape ``batch_shape``) and return a floating point tensor of of
        shape ``batch_shape + (F,)`` for some number of features ``F``. This
        will be called internally during inference.
    :param dict config: Optional config dict. See keyword arguments to
        :func:`~pyroed.oed.thompson_sample` for details.
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
            config=config,
        )

    # Perform OED via Thompson sampling.
    design_set = thompson_sample(
        schema,
        constraints,
        feature_blocks,
        gibbs_blocks,
        experiment,
        design_size=design_size,
        feature_fn=feature_fn,
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
    :param torch.Tensor new_responses: A tensor of the measured responses of sequences.
    :param torch.Tensor new_batch_ids: An optional tensor of batch ids.
    :returns: A concatenated experiment.
    :rtype: dict
    """
    # If unspecified, simply create a new single batch id.
    if new_batch_ids is None:
        new_batch_ids = experiment["batch_ids"].new_full(
            new_responses.shape, experiment["batch_ids"].max().item() + 1
        )

    # Validate.
    if __debug__:
        validate(schema, experiment=experiment)
    assert len(new_responses) == len(new_sequences)
    assert len(new_batch_ids) == len(new_sequences)

    # Concatenate the dictionaries.
    new_experiment = {
        "sequences": new_sequences,
        "responses": new_responses,
        "batch_ids": new_batch_ids,
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
