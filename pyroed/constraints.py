from abc import ABC, abstractmethod
from typing import Optional

import torch

from .typing import Schema


class Constraint(ABC):
    @abstractmethod
    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AllDifferent(Constraint):
    def __init__(self, *names):
        self.names = names

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        name_to_int = {name: i for i, name in enumerate(schema)}
        ps = [name_to_int[name] for name in self.names]
        ok = torch.tensor(True)
        for i, p1 in enumerate(ps):
            for p2 in ps[:i]:
                ok = ok & (choices[..., p1] != choices[..., p2])
        return ok


class TakesValue(Constraint):
    def __init__(self, name: str, value: Optional[str]):
        self.name = name
        self.value = value

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        for k, (name, values) in enumerate(schema.items()):
            if name == self.name:
                v = values.index(self.value)
                return choices[..., k] == v
        raise ValueError


class IfThen(Constraint):
    def __init__(self, lhs: Constraint, rhs: Constraint):
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        lhs = self.lhs(schema, choices)
        rhs = self.rhs(schema, choices)
        return rhs | ~lhs


class Iff(Constraint):
    def __init__(self, lhs: Constraint, rhs: Constraint):
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        lhs = self.lhs(schema, choices)
        rhs = self.rhs(schema, choices)
        return lhs == rhs
