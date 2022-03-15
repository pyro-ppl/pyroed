from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

from .typing import Schema


class Constraint(ABC):
    """
    Abstract base class for constraints
    """

    @abstractmethod
    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TakesValue(Constraint):
    """
    Constrains a site to take a fixed value.
    """

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


class TakesValues(Constraint):
    """
    Constrains a site to take one of a set of values.
    """

    def __init__(self, name: str, *values: Optional[str]):
        super().__init__()
        self.name = name
        self.values = values
        self.schema: Optional[Schema] = None
        self.index: Optional[int] = None
        self.mask: Optional[torch.Tensor] = None

    def __str__(self):
        args = ", ".join(map(repr, (self.name,) + self.values))
        return f"TakesValues({args})"

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        # Lazily build tensor indices.
        if schema is not self.schema:
            assert self.name in schema
            for index, (name, values) in enumerate(schema.items()):
                if name == self.name:
                    self.schema = schema
                    self.index = index
                    self.mask = torch.zeros(len(values), dtype=torch.bool)
                    assert set(self.values).issubset(values)
                    for i, value in enumerate(values):
                        if value in self.values:
                            self.mask[i] = True
                    break
        assert self.mask is not None
        assert self.index is not None

        # Compute the constraint.
        return self.mask[choices[..., self.index]]


class AllDifferent(Constraint):
    """
    Constrains a set of sites to all have distinct values.
    """

    def __init__(self, *names: str):
        super().__init__()
        self.names = names
        self.schema: Optional[Schema] = None
        self.name_to_int: Optional[Dict[str, int]] = None
        self.standardize: Optional[torch.Tensor] = None

    def __str__(self):
        return "AllDifferent({})".format(", ".join(map(repr, self.names)))

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        # Lazily build tensor indices.
        if schema is not self.schema:
            self.schema = schema
            self.name_to_int = {name: i for i, name in enumerate(schema)}
            standard = list(set().union(*(schema[n] for n in self.names)))
            standard.sort(key=lambda x: (0, "") if x is None else (1, x))
            self.standardize = choices.new_empty(
                len(self.names), max(len(schema[n]) for n in self.names)
            )
            for i, name in enumerate(self.names):
                for j, value in enumerate(schema[name]):
                    self.standardize[i, j] = standard.index(value)
        assert self.name_to_int is not None
        assert self.standardize is not None

        # Compute the constraint.
        ps = [self.name_to_int[name] for name in self.names]
        ok = torch.tensor(True)
        for i, p1 in enumerate(ps):
            c1 = self.standardize[i][choices[..., p1]]
            for j, p2 in enumerate(ps[:i]):
                c2 = self.standardize[j][choices[..., p2]]
                ok = ok & (c1 != c2)
        return ok


class Not(Constraint):
    """
    Negates a constraints.
    """

    def __init__(self, arg: Constraint):
        super().__init__()
        self.arg = arg

    def __str__(self):
        return f"Not({self.arg})"

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        arg = self.arg(schema, choices)
        return ~arg


class And(Constraint):
    """
    Conjoins two constraints.
    """

    def __init__(self, lhs: Constraint, rhs: Constraint):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"And({self.lhs}, {self.rhs})"

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        lhs = self.lhs(schema, choices)
        rhs = self.rhs(schema, choices)
        return rhs & lhs


class Or(Constraint):
    """
    Disjoins two constraints.
    """

    def __init__(self, lhs: Constraint, rhs: Constraint):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"Or({self.lhs}, {self.rhs})"

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        lhs = self.lhs(schema, choices)
        rhs = self.rhs(schema, choices)
        return rhs | lhs


class Xor(Constraint):
    """
    Exclusive or among constraints. Equivalent to ``Not(Iff(lhs, rhs))``.
    """

    def __init__(self, lhs: Constraint, rhs: Constraint):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"Xor({self.lhs}, {self.rhs})"

    def __call__(self, schema: Schema, choices: torch.Tensor) -> torch.Tensor:
        lhs = self.lhs(schema, choices)
        rhs = self.rhs(schema, choices)
        return rhs ^ lhs


class IfThen(Constraint):
    """
    Conditional between constraints.
    """

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
    """
    Equality among constraints.
    """

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
