from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch

# Cannot use OrderedDict yet https://stackoverflow.com/questions/41207128
Schema = Dict[str, List[Optional[str]]]
Coefs = Dict[Tuple[str, ...], torch.Tensor]
Blocks = List[List[str]]
Constraints = List[Callable]


def validate(
    schema: Schema,
    *,
    constraints: Optional[Constraints] = None,
    feature_blocks: Optional[Blocks] = None,
    gibbs_blocks: Optional[Blocks] = None,
    experiment: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """
    Validates a Pyroed problem specification.

    :param OrderedDict schema: A schema dict.
    :param list constraints: An optional list of constraints.
    :param list feature_blocks: An optional list of choice blocks for linear
        regression.
    :param list gibbs_blocks: An optional list of choice blocks for Gibbs
        sampling.
    :param dict experiment: An optional dict containing all old experiment data.
    """
    from .constraints import Constraint  # avoid import cycle

    # Validate schema.
    assert isinstance(schema, OrderedDict)
    for name, values in schema.items():
        assert isinstance(name, str)
        assert isinstance(values, list)
        assert values
        for value in values:
            assert value is None or isinstance(value, str)

    # Validate constraints.
    if constraints is not None:
        assert isinstance(constraints, list)
        for constraint in constraints:
            assert isinstance(constraint, Constraint)

    # Validate feature_blocks.
    if feature_blocks is not None:
        assert isinstance(feature_blocks, list)
        for block in feature_blocks:
            assert isinstance(block, list)
            for col in block:
                assert col in schema
        assert len({tuple(f) for f in feature_blocks}) == len(
            feature_blocks
        ), "duplicate feature_blocks"

    # Validate gibbs_blocks.
    if gibbs_blocks is not None:
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
        allowed_keys = {"sequences", "batch_ids", "responses"}
        required_keys = {"sequences", "batch_ids"}
        assert allowed_keys.issuperset(experiment)
        assert required_keys.issubset(experiment)

        sequences = experiment["sequences"]
        assert isinstance(sequences, torch.Tensor)
        assert sequences.dtype == torch.long
        assert sequences.dim() == 2
        assert sequences.shape[-1] == len(schema)

        batch_id = experiment["batch_ids"]
        assert isinstance(batch_id, torch.Tensor)
        assert batch_id.dtype == torch.long
        assert batch_id.shape == sequences.shape[:1]

        response = experiment.get("responses")
        if response is not None:
            assert isinstance(response, torch.Tensor)
            assert torch.is_floating_point(response)
            assert response.shape == sequences.shape[:1]
            assert 0 <= response.min()
            assert response.max() <= 1
