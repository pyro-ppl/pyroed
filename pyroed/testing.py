import warnings
from typing import Dict, Tuple

import pyro.poutine as poutine
import torch

from .models import model
from .typing import Blocks, Schema, validate


@torch.no_grad()
def generate_fake_data(
    schema: Schema,
    feature_blocks: Blocks,
    sequences_per_batch: int,
    num_batches: int = 1,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Generates a fake dataset for testing.

    :param OrderedDict schema: A schema dict.
    :param list constraints: A list of constraints.
    :param list feature_blocks: A list of choice blocks for linear regression.
    :param int sequences_per_batch: The number of sequences per experiment
        batch.
    :param int num_batches: The number of experiment batches.
    :returns: A pair ``(truth, experiment)``, where ``truth`` is a dict of
        true values of latent variables (regression coefficients, etc.), and
        ``experiment`` is a standard experiment dict.
    :rtype: tuple
    """
    B = num_batches
    N = sequences_per_batch * B
    experiment: Dict[str, torch.Tensor] = {}

    # Work around irrelevant PyTorch interface change warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "floor_divide is deprecated", UserWarning)
        experiment["batch_ids"] = torch.arange(N) // sequences_per_batch

    experiment["sequences"] = torch.stack(
        [torch.randint(0, len(choices), (N,)) for choices in schema.values()], dim=-1
    )
    trace = poutine.trace(model).get_trace(schema, feature_blocks, experiment)
    truth: Dict[str, torch.Tensor] = {
        name: site["value"].detach()
        for name, site in trace.nodes.items()
        if site["type"] == "sample" and not site["is_observed"]
        if type(site["fn"]).__name__ != "_Subsample"
        if name != "batch_response"  # shape varies in time
    }
    experiment["responses"] = trace.nodes["responses"]["value"].detach()
    if __debug__:
        validate(schema, feature_blocks=feature_blocks, experiment=experiment)

    return truth, experiment
