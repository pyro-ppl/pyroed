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
        super().__init__()
        self.names = names

    def __str__(self):
        return "AllDifferent({})".format(", ".join(map(repr, self.names)))

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
        super().__init__()
        self.name = name
        self.value = value

    def __str__(self):
        return f"TakesValue({repr(self.name)}, {repr(self.value)})"

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        for k, (name, values) in enumerate(schema.items()):
            if name == self.name:
                if self.value not in values:
                    raise ValueError(
                        f"In constraint {self}: "
                        f"{repr(self.value)} not found in schema[repr(name)]"
                    )
                v = values.index(self.value)
                return choices[..., k] == v
        raise ValueError(
            f"In constraint {self}: {repr(self.value)} not found in schema"
        )


class IfThen(Constraint):
    def __init__(self, lhs: Constraint, rhs: Constraint):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"IfThen({self.lhs}, {self.rhs})"

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        lhs = self.lhs(schema, choices)
        rhs = self.rhs(schema, choices)
        return rhs | ~lhs


class Iff(Constraint):
    def __init__(self, lhs: Constraint, rhs: Constraint):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"IfThen({self.lhs}, {self.rhs})"

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        lhs = self.lhs(schema, choices)
        rhs = self.rhs(schema, choices)
        return lhs == rhs
