from collections import OrderedDict
from typing import List, Optional

from pyroed.api import encode_design
from pyroed.constraints import (
    AllDifferent,
    And,
    Iff,
    IfThen,
    Not,
    Or,
    TakesValue,
    TakesValues,
    Xor,
)
from pyroed.typing import Schema


def stringify(bools: List[bool]) -> str:
    return "".join("1" if x else "0" for x in bools)


def test_immune_sequence():
    SCHEMA: Schema = OrderedDict()
    SCHEMA["Protein 1"] = ["Prot1", "Prot2", None]
    SCHEMA["Protein 2"] = ["Prot3", "HLA1", "HLA2", "HLA3", "HLA4", None]
    SCHEMA["Signalling Pep"] = ["Sig1", "Sig2", None]
    SCHEMA["EP"] = [f"Ep{i}" for i in range(1, 10 + 1)]
    SCHEMA["EP"].append(None)
    SCHEMA["Linker"] = ["Link1", None]
    SCHEMA["Internal"] = ["Int1", "Int2", "Int3", "Int3", None]
    SCHEMA["2A-1"] = ["twoa1", "twoa2", None]
    SCHEMA["2A-2"] = ["twoa3", "twoa4", None]
    SCHEMA["2A-3"] = [f"twoa{i}" for i in range(1, 7 + 1)]

    CONSTRAINTS = [
        AllDifferent("2A-1", "2A-2", "2A-3"),
        Iff(TakesValue("Protein 1", None), TakesValue("2A-1", None)),
        Iff(TakesValue("Signalling Pep", None), TakesValue("EP", None)),
        Iff(TakesValue("EP", None), TakesValue("Linker", None)),
        IfThen(TakesValue("Protein 2", None), TakesValue("Internal", None)),
        Iff(TakesValue("Protein 2", "Prot3"), TakesValue("2A-2", None)),
    ]

    design: List[List[Optional[str]]] = [
        ["Prot1", "Prot3", "Sig1", "Ep1", "Link1", "Int1", "twoa1", None, "twoa2"],
        ["Prot1", "Prot3", "Sig1", "Ep1", "Link1", "Int1", "twoa1", None, "twoa1"],
        [None, "Prot3", "Sig1", "Ep1", "Link1", "Int1", "twoa1", None, "twoa2"],
        ["Prot1", "Prot3", "Sig1", None, None, None, "twoa1", None, "twoa2"],
        ["Prot1", "Prot3", "Sig1", "Ep1", None, "Int1", "twoa1", None, "twoa2"],
        ["Prot1", None, "Sig1", "Ep1", "Link1", "Int1", "twoa1", "twoa4", "twoa2"],
        ["Prot1", "Prot3", "Sig1", "Ep1", "Link1", "Int1", "twoa1", "twoa4", "twoa2"],
    ]

    sequences = encode_design(SCHEMA, design)
    actual = [c(SCHEMA, sequences).tolist() for c in CONSTRAINTS]
    assert actual[0] == [True, False, True, True, True, True, True]
    assert actual[1] == [True, True, False, True, True, True, True]
    assert actual[2] == [True, True, True, False, True, True, True]
    assert actual[3] == [True, True, True, True, False, True, True]
    assert actual[4] == [True, True, True, True, True, False, True]
    assert actual[5] == [True, True, True, True, True, True, False]


def test_takes_value():
    SCHEMA = OrderedDict()
    SCHEMA["foo"] = ["a", "b", "c", None]
    SCHEMA["bar"] = ["a", "b", None]

    CONSTRAINTS = [
        TakesValue("foo", "a"),
        TakesValue("foo", "b"),
        TakesValue("foo", "c"),
        TakesValue("foo", None),
        TakesValue("bar", "a"),
        TakesValue("bar", "b"),
        TakesValue("bar", None),
    ]

    design: List[List[Optional[str]]] = [
        ["a", "a"],
        ["a", "b"],
        ["a", None],
        ["b", "a"],
        ["b", "b"],
        ["b", None],
        ["c", "a"],
        ["c", "b"],
        ["c", None],
        [None, "a"],
        [None, "b"],
        [None, None],
    ]

    sequences = encode_design(SCHEMA, design)
    actual = [c(SCHEMA, sequences).tolist() for c in CONSTRAINTS]
    assert stringify(actual[0]) == "111000000000"
    assert stringify(actual[1]) == "000111000000"
    assert stringify(actual[2]) == "000000111000"
    assert stringify(actual[3]) == "000000000111"
    assert stringify(actual[4]) == "100100100100"
    assert stringify(actual[5]) == "010010010010"
    assert stringify(actual[6]) == "001001001001"


def test_takes_values():
    SCHEMA = OrderedDict()
    SCHEMA["foo"] = ["a", "b", "c", None]
    SCHEMA["bar"] = ["a", "b", None]

    CONSTRAINTS = [
        TakesValues("foo", "a"),
        TakesValues("foo", "b", "c"),
        TakesValues("foo", "a", None),
        TakesValues("bar", "a", "b", None),
        TakesValues("bar", "b"),
        TakesValues("bar"),
    ]

    design: List[List[Optional[str]]] = [
        ["a", "a"],
        ["a", "b"],
        ["a", None],
        ["b", "a"],
        ["b", "b"],
        ["b", None],
        ["c", "a"],
        ["c", "b"],
        ["c", None],
        [None, "a"],
        [None, "b"],
        [None, None],
    ]

    sequences = encode_design(SCHEMA, design)
    actual = [c(SCHEMA, sequences).tolist() for c in CONSTRAINTS]
    assert stringify(actual[0]) == "111000000000"
    assert stringify(actual[1]) == "000111111000"
    assert stringify(actual[2]) == "111000000111"
    assert stringify(actual[3]) == "111111111111"
    assert stringify(actual[4]) == "010010010010"
    assert stringify(actual[5]) == "000000000000"


def test_logic():
    SCHEMA = OrderedDict()
    SCHEMA["foo"] = ["a", None]
    SCHEMA["bar"] = ["a", None]

    foo = TakesValue("foo", "a")
    bar = TakesValue("bar", "a")
    CONSTRAINTS = [
        foo,
        bar,
        Not(foo),
        Not(bar),
        And(foo, bar),
        Or(foo, bar),
        Xor(foo, bar),
        Iff(foo, bar),
        IfThen(foo, bar),
    ]

    design: List[List[Optional[str]]] = [
        ["a", "a"],
        ["a", None],
        [None, "a"],
        [None, None],
    ]

    sequences = encode_design(SCHEMA, design)
    actual = [c(SCHEMA, sequences).tolist() for c in CONSTRAINTS]
    assert stringify(actual[0]) == "1100"
    assert stringify(actual[1]) == "1010"
    assert stringify(actual[2]) == "0011"
    assert stringify(actual[3]) == "0101"
    assert stringify(actual[4]) == "1000"
    assert stringify(actual[5]) == "1110"
    assert stringify(actual[6]) == "0110"
    assert stringify(actual[7]) == "1001"
    assert stringify(actual[8]) == "1011"
