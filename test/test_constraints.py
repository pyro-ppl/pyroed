from collections import OrderedDict

from pyroed.api import encode_design
from pyroed.constraints import AllDifferent, Iff, IfThen, TakesValue


def test_immune_sequence():
    SCHEMA = OrderedDict()
    SCHEMA["Protein 1"] = ["Prot1", "Prot2", None]
    SCHEMA["Protein 2"] = ["Prot3", "HLA1", "HLA2", "HLA3", "HLA4", None]
    SCHEMA["Signalling Pep"] = ["Sig1", "Sig2", None]
    SCHEMA["EP"] = [f"Ep{i}" for i in range(1, 10 + 1)] + [None]
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

    design = [
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
