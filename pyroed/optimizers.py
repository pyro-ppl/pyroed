import math
import operator
from collections import OrderedDict
from functools import reduce
from typing import Callable, List, Optional

import pyro.distributions as dist
import torch


def optimize_simulated_annealing(
    schema: OrderedDict,
    constraints: list,
    gibbs_blocks: Optional[List[List[str]]],
    response_fn: Callable,
    coefs: dict,
    *,
    temperature_schedule: torch.Tensor,
    log_every=100,
) -> torch.Tensor:
    """
    Optimize by simulated annealing.
    """
    # Set up problem shape.
    P = len(schema)
    num_categories = torch.tensor([len(v) for v in schema.values()])
    bounds = dist.constraints.integer_interval(0, num_categories)
    if gibbs_blocks is None:
        gibbs_blocks = [[name] for name in schema]  # single site
    assert set(sum(gibbs_blocks, [])) == set(schema), "invalid gibbs blocks"
    name_to_int = {name: i for i, name in enumerate(schema)}
    int_blocks = [[name_to_int[name] for name in block] for block in gibbs_blocks]

    def constraint_fn(seq):
        return reduce(operator.and_, (c(seq) for c in constraints), True)

    # Initialize to a single random uniform feasible state.
    for i in range(10000):
        state = torch.stack(
            [torch.randint(0, Cp, ()) for Cp in num_categories.tolist()]
        )
        assert bounds.check(state).all()
        if constraint_fn(schema, state):
            break
    if not constraint_fn(schema, state):
        raise ValueError("Failed to find a feasible initial state")

    # Anneal, recording the best state.
    best_state = None
    best_response = -math.inf
    for step, temperature in enumerate(temperature_schedule):
        # Choose a random Gibbs block.
        b = int(torch.randint(0, len(gibbs_blocks), ()))
        block = int_blocks[b]
        Cs = tuple(int(num_categories[p]) for p in block)

        # Create a cartesian product over choices within the block.
        nbhd = state.expand(Cs + (P,)).clone()
        for i, (p, C) in enumerate(zip(block, Cs)):
            nbhd[..., p] = torch.arange(C).reshape((-1,) + (1,) * i)
        nbhd = nbhd.reshape(-1, P)

        # Restrict to feasible states.
        ok = constraint_fn(schema, nbhd)
        if ok is not True:
            nbhd = nbhd[ok]
        assert bounds.check(nbhd).all()

        # Randomly sample variables in the block wrt an annealed response.
        response = response_fn(coefs, nbhd)
        assert response.dim() == 1
        choice = int(dist.Categorical(logits=response / temperature).sample())

        # Update state with choices within the block.
        for p, C in zip(block, Cs):
            state[p] = choice % C
            choice //= C
        assert bounds.check(state).all()

        # Save the best response.
        current_response = response[choice].item()
        if current_response > best_response:
            best_state = state.clone()
            best_response = current_response
        if log_every and step % log_every == 0:
            print(
                f"sa step {step} temp={temperature:0.3g} response={current_response:0.6g}"
            )

    return best_state
