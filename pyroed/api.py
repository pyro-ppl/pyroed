"""
Pyroed's high-level interface includes a design language and a set of
functions to operate on Python data structures.

The **design language** allows you to specify a problem by defining a
``SCHEMA``, a list ``CONSTRAINTS`` of :class:`~pyroed.constraints.Constraint`
objects, a list ``FEATURE_BLOCKS`` defining cross features, and a list
``GIBBS_BLOCKS`` defining groups of features that are related to each other.
The examples in this module will use the following model specification::

    SCHEMA = OrderedDict()
    SCHEMA["aa1"] = ["P", "L", None]
    SCHEMA["aa2"] = ["N", "Y",  "T", None]
    SCHEMA["aa3"] = ["R", "S"]

    CONSTRAINTS = [Not(And(TakesValue("aa1", None), TakesValue("aa2", None)))]

    FEATURE_BLOCKS = [["aa1"], ["aa2"], ["aa3"], ["aa1", "aa2"], ["aa2", "aa3"]]

    GIBBS_BLOCKS = [["aa1", "aa2"], ["aa2", "aa3"]]

After declaring the design space, we can progressively gather data into an
``experiment`` dict by using the functions in this module and by experimentally
measuring sequences.

- :func:`encode_design` and :func:`decode_design` convert between
  text-representations of designs like ``[["P", "N", "R"], ["P", "N", "S"]]``
  and PyTorch representations of designs like
  ``torch.tensor([[0, 0, 0], [0, 0, 1]])``.
- :func:`start_experiment` initializes an experiment dict,
- :func:`get_next_design` suggests a next set of sequences to test, and
- :func:`update_experiment` updates an experiment dict with measured responses.

Note that :func:`get_next_design` merely retuns suggested sequences; you can
ignore these suggestions or measure a different set of sequences if you want.
For example if some of your measurements are lost due to technical reasons, you
can simply pass a subset of the suggested design back to
:func:`update_experiment`.
"""

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

    Example::

        SCHEMA = OrderedDict()
        SCHEMA["aa1"] = ["P", "L", None]
        SCHEMA["aa2"] = ["N", "Y",  "T", None]
        SCHEMA["aa3"] = ["R", "S"]

        design = [
            ["P", "N", "R"],
            ["P", "N", "S"],
            [None, "N", "R"],
            ["P", None, "R"],
        ]
        sequences = encode_design(SCHEMA, design)
        print(sequences)
        # torch.tensor([[0, 0, 0], [0, 0, 1], [2, 0, 0], [0, 3, 0]])

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

    Example::

        SCHEMA = OrderedDict()
        SCHEMA["aa1"] = ["P", "L", None]
        SCHEMA["aa2"] = ["N", "Y",  "T", None]
        SCHEMA["aa3"] = ["R", "S"]

        sequences = torch.tensor([[0, 0, 0], [0, 0, 1], [2, 0, 0]])
        design = decode_design(SCHEMA, sequences)
        print(design)
        # [["P", "N", "R"], ["P", "N", "S"], [None, "N", "R"]]

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

    Example::

        SCHEMA = OrderedDict()
        SCHEMA["aa1"] = ["P", "L", None]
        SCHEMA["aa2"] = ["N", "Y",  "T", None]
        SCHEMA["aa3"] = ["R", "S"]

        sequences = torch.tensor([[0, 0, 0], [0, 0, 1], [2, 0, 0]])
        responses = torch.tensor([0.1, 0.4, 0.5])

        experiment = start_experiment(SCHEMA, sequences, responses)

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

    Under the hood this runs :func:`~pyroed.oed.thompson_sample`, which
    performs Bayesian inference via either variational inference
    :func:`~pyroed.inference.fit_svi` or MCMC
    :func:`~pyroed.inference.fit_mcmc` and performs optimization via
    :func:`~pyroed.optimizers.optimize_simulated_annealing`. These algorithms
    can be tuned through the ``config`` dict.

    Example::

        # Initialize experiment.
        sequences = encode_design(SCHEMA, [
            ["P", "N", "R"],
            ["P", "N", "S"],
            [None, "N", "R"],
            ["P", None, "R"],
        ])
        print(sequences)
        # torch.tensor([[0, 0, 0], [0, 0, 1], [2, 0, 0], [0, 3, 0]])
        experiment = {
            "sequences": sequences,
            "responses": torch.tensor([0.1, 0.4, 0.5, 0.2]),
            "batch_ids": torch.tensor([0, 0, 1, 1]),
        }

        # Run Bayesian optimization to get the next sequences to measure.
        new_sequences = get_next_design(
            SCHEMA, CONSTRAINTS, FEATURE_BLOCKS, GIBBS_BLOCKS,
            experiment, design_size=2,
        )
        print(new_sequences)
        # torch.tensor([[1, 1, 1], [1, 2, 0]])
        print(decode_design(SCHEMA, new_sequences))
        # [["L", "Y", "S"], ["L", T", "R"]]

    :param OrderedDict schema: A schema dict.
    :param list constraints: A list of zero or more
        :class:`~pyroed.constraints.Constraint` objects.
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

    Note this does not modify its arguments; you must capture the result::

        experiment = update_experiment(
            SCHEMA, experiment, new_sequences, new_responses, new_batch_ids
        )

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
