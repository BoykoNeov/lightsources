"""Parameters, reference library and lamp presets."""

import numpy as np
import pytest

from lss.accuracy import references
from lss.lamp import lamp as lamps
from lss.lamp.profile import COMPONENTS, EmissionProfile, Zone
from lss.params import Parameter
from lss.spectral.spectrum import Spectrum


def test_physical_parameter_needs_source_and_empirical_needs_reason():
    Parameter(2856, "K", source="CIE 15:2018")
    Parameter(0.3, "", empirical=True, reason="coil cavity factor not yet modelled")
    with pytest.raises(ValueError):
        Parameter(2856, "K")
    with pytest.raises(ValueError):
        Parameter(0.3, "", empirical=True)
    with pytest.raises(ValueError):
        Parameter(5.0, "K", source="x", bounds=(0.0, 1.0))


def test_every_reference_loads_with_provenance():
    lib = references.library()
    assert {"cie-a", "cie-fl11", "cie-hp1", "cie-led-b3"} <= set(lib)
    for ref in lib.values():
        assert ref.citation and ref.data_version and ref.uncertainty and ref.description
        assert ref.family in {"incandescent", "fluorescent", "hid", "led"}
        assert ref.includes <= COMPONENTS
        lo, hi = ref.valid_range
        if ref.scored_at_nodes:  # an uneven measured table: its nodes are its sampling
            assert ref.sampling_nm is None and ref.nodes.size == ref.node_values.size
            assert ref.nodes[0] == lo and ref.nodes[-1] == hi and np.all(np.diff(ref.nodes) > 0)
        else:
            assert ref.sampling_nm > 0
            assert lo >= 300 and hi <= 830
        assert lo < hi
        assert ref.spectrum.total() > 0


def test_cie_tables_are_marked_coarse():
    # 5 nm tables cannot check features of a few nm (HPS self-reversal).
    assert references.get("cie-hp1").sampling_nm == 5


@pytest.mark.parametrize("preset", lamps.preset_ids())
def test_presets_generate_every_level(preset):
    lamp = lamps.load(preset)
    assert lamp.reference in references.library()
    for level in lamp.models:
        profile = lamp.generate(level)
        assert profile.metadata["level"] == level
        assert profile.metadata["data_versions"]
        assert profile.spectrum.total() > 0
        for p in profile.metadata["params"].values():
            assert p.get("source") or p.get("reason")


def test_empirical_count_covers_only_parameters_a_level_uses():
    lamp = lamps.lamp_from_dict({
        "id": "t", "family": "incandescent", "reference": "cie-a",
        "physical": {
            "filament_temperature": {"value": 2800, "unit": "K", "empirical": True, "reason": "fitted"},
            "coil_factor": {"value": 0.3, "unit": "", "empirical": True, "reason": "unused here"},
        },
        "models": {0: {"generator": "reference"}, 1: {"generator": "planck"}},
    })
    assert lamp.generate(0).empirical_count == 0
    assert lamp.generate(1).empirical_count == 1


def test_reference_shortcut_builds_level_zero_lamp():
    lamp = lamps.load("ref:cie-led-v1")
    profile = lamp.generate(0)
    assert profile.includes == {"phosphor"}
    assert profile.metadata["valid_range_nm"] == (380.0, 780.0)


def test_unknown_names_fail_clearly():
    with pytest.raises(KeyError, match="presets"):
        lamps.load("no_such_lamp")
    with pytest.raises(KeyError, match="lss refs"):
        lamps.load("ref:nope")
    with pytest.raises(KeyError, match="no level 7"):
        lamps.load("incandescent_cie_a").generate(7)


def test_profile_rejects_unknown_component():
    with pytest.raises(ValueError):
        EmissionProfile((Zone(Spectrum()),), includes=frozenset({"ballast"}))
