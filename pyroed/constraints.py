from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

from .typing import Schema

Choices = Dict[str, torch.Tensor]


class Constraint(ABC):
    @abstractmethod
    def __call__(self, schema: Schema, choices: Choices) -> torch.Tensor:
        raise NotImplementedError


class AllDifferent(Constraint):
    def __init__(self, *names):
        self.names = names

    def __call__(self, schema: Schema, choices: Choices) -> torch.Tensor:
        ok = torch.tensor(True)
        for i, a in enumerate(self.names):
            for b in self.names[:i]:
                ok = ok & (choices[a] != choices[b])
        return ok


class TakesValue(Constraint):
    def __init__(self, name: str, value: Optional[str]):
        self.name = name
        self.value = value

    def __call__(self, schema: Schema, choices: Choices) -> torch.Tensor:
        i = schema[self.name].index(self.value)
        return choices[self.name] == i


class IfThen(Constraint):
    def __init__(self, lhs: Constraint, rhs: Constraint):
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self, schema: Schema, choices: Choices) -> torch.Tensor:
        lhs = self.lhs(schema, choices)
        rhs = self.rhs(schema, choices)
        return rhs | ~lhs


class Iff(Constraint):
    def __init__(self, lhs: Constraint, rhs: Constraint):
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self, schema: Schema, choices: Choices) -> torch.Tensor:
        lhs = self.lhs(schema, choices)
        rhs = self.rhs(schema, choices)
        return lhs == rhs
