import warnings

import pyro.poutine as poutine
import torch

from .models import model
from .typing import Features, Schema


@torch.no_grad()
def generate_fake_data(
    schema: Schema,
    features: Features,
    sequences_per_batch: int,
    num_batches: int = 1,
):
    B = num_batches
    N = sequences_per_batch * B
    experiment = {}

    # Work around irrelevant PyTorch interface change warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "floor_divide is deprecated", UserWarning)
        experiment["batch_id"] = torch.arange(N) // sequences_per_batch

    experiment["sequences"] = torch.stack(
        [torch.randint(0, len(choices), (N,)) for choices in schema.values()], dim=-1
    )
    experiment["response"] = None
    trace = poutine.trace(model).get_trace(schema, features, experiment)
    truth = {
        name: site["value"].detach()
        for name, site in trace.nodes.items()
        if site["type"] == "sample" and not site["is_observed"]
        if type(site["fn"]).__name__ != "_Subsample"
        if name != "batch_response"  # shape varies in time
    }
    experiment["response"] = trace.nodes["response"]["value"].detach()
    return truth, experiment
