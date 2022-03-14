import operator
from functools import reduce

import pyro.distributions as dist
import torch

from .models import linear_response
from .typing import Blocks, Constraints, Schema


@torch.no_grad()
def optimize_simulated_annealing(
    schema: Schema,
    constraints: Constraints,
    gibbs_blocks: Blocks,
    coefs: dict,
    *,
    temperature_schedule: torch.Tensor,
    max_tries=10000,
    log_every=100,
) -> torch.Tensor:
    """
    Optimize by simulated annealing.
    """
    # Set up problem shape.
    P = len(schema)
    num_categories = torch.tensor([len(v) for v in schema.values()])
    bounds = dist.constraints.integer_interval(0, num_categories)
    assert set(sum(gibbs_blocks, [])) == set(schema), "invalid gibbs blocks"
    name_to_int = {name: i for i, name in enumerate(schema)}
    int_blocks = [[name_to_int[name] for name in block] for block in gibbs_blocks]

    def constraint_fn(seq):
        if not constraints:
            return True
        return reduce(operator.and_, (c(schema, seq) for c in constraints))

    # Initialize to a single random uniform feasible state.
    for i in range(max_tries):
        state = torch.stack(
            [torch.randint(0, Cp, ()) for Cp in num_categories.tolist()]
        )
        assert bounds.check(state).all()
        if constraint_fn(state):
            break
    if not constraint_fn(state):
        raise ValueError("Failed to find a feasible initial state")
    best_state = state
    best_logits = float(linear_response(schema, coefs, state))

    # Anneal, recording the best state.
    for step, temperature in enumerate(temperature_schedule):
        # Choose a random Gibbs block.
        b = int(torch.randint(0, len(gibbs_blocks), ()))
        block = int_blocks[b]
        Cs = [int(num_categories[p]) for p in block]

        # Create a cartesian product over choices within the block.
        nbhd = state.expand(tuple(reversed(Cs)) + (P,)).clone()
        for i, (p, C) in enumerate(zip(block, Cs)):
            nbhd[..., p] = torch.arange(C).reshape((-1,) + (1,) * i)
        nbhd = nbhd.reshape(-1, P)

        # Restrict to feasible states.
        ok = constraint_fn(nbhd)
        if ok is not True:
            nbhd = nbhd[ok]
        assert bounds.check(nbhd).all()

        # Randomly sample variables in the block wrt an annealed logits.
        logits = linear_response(schema, coefs, nbhd)
        assert logits.dim() == 1
        choice = dist.Categorical(logits=logits / temperature).sample()
        state[:] = nbhd[choice]
        assert bounds.check(state).all()
        assert constraint_fn(state)

        # Save the best response.
        current_logits = float(logits[choice])
        if current_logits > best_logits:
            best_state = state.clone()
            best_logits = current_logits
        if log_every and step % log_every == 0:
            print(
                f"sa step {step} temp={temperature:0.3g} "
                f"logits={current_logits:0.6g}"
            )

    return best_state
