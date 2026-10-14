"""Colour checks (design §12): CIE illuminants and agreement with colour-science."""

import warnings

import numpy as np
import pytest

from lss.spectral import colorimetry as cm
from lss.spectral.continuum import PlanckContinuum, TabulatedContinuum
from lss.spectral.grids import analysis_grid, render_grid
from lss.spectral.spectrum import Spectrum
from lss.spectral.units import C2_CIE_ILLUMINANT_A

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import colour


def cie_table(key):
    sd = colour.SDS_ILLUMINANTS[key]
    return Spectrum.of(TabulatedContinuum(sd.wavelengths, sd.values)).bin(analysis_grid())


def illuminant_a_definition():
    # CIE 15:2018: Planck at 2848 K with the legacy c2 = 1.435e-2 m·K
    # (identical in shape to 2856 K with today's c2).
    return Spectrum.of(PlanckContinuum(2848.0, c2=C2_CIE_ILLUMINANT_A)).bin(analysis_grid())


@pytest.mark.parametrize("method", ["Ohno 2013", "Robertson 1968"])
def test_illuminant_a_cct(method):
    # CCT methods differ by about 1 K; the named value is 2856 K.
    for binned in (illuminant_a_definition(), cie_table("A")):
        cct, duv = cm.cct_duv(binned, method)
        assert cct == pytest.approx(2856.0, abs=2.0)
        assert abs(duv) < 1e-4


def test_illuminant_a_renders_perfectly():
    cr = cm.colour_rendering(illuminant_a_definition())
    assert cr.ra == pytest.approx(100.0, abs=0.05)
    assert cr.r9 == pytest.approx(100.0, abs=0.05)
    assert cr.rf == pytest.approx(100.0, abs=0.05)
    assert cr.rg == pytest.approx(100.0, abs=0.05)


def test_d65_chromaticity():
    np.testing.assert_allclose(cm.xy(cie_table("D65")), [0.3127, 0.3290], atol=2e-4)


def test_equal_energy_chromaticity():
    flat = Spectrum.of(TabulatedContinuum(np.array([300.0, 900.0]), np.array([1.0, 1.0])))
    np.testing.assert_allclose(cm.xy(flat.bin(analysis_grid())), [1 / 3, 1 / 3], atol=1e-4)


@pytest.mark.parametrize("key", ["A", "D65", "FL2", "FL11", "HP1", "LED-B3"])
def test_wrappers_agree_with_colour_science(key):
    binned = cie_table(key)
    sd = cm.to_sd(binned)
    np.testing.assert_allclose(
        cm.tristimulus(binned), colour.sd_to_XYZ(sd, k=1, method="Integration"), rtol=1e-12
    )
    cct, duv = cm.cct_duv(binned)
    ref = colour.uv_to_CCT(colour.xy_to_UCS_uv(colour.XYZ_to_xy(colour.sd_to_XYZ(sd, k=1, method="Integration"))))
    assert cct == pytest.approx(ref[0], rel=1e-9)
    assert duv == pytest.approx(ref[1], abs=1e-9)
    # Against colour-science's own path from the 5 nm table. We read the table as
    # piecewise linear, colour-science interpolates it smoothly; for spiky lamps
    # (fluorescent, HPS) a 5 nm table cannot tell the two readings apart, and they
    # differ by up to ~2.5e-4 in xy. Smooth spectra agree to 2e-4.
    spiky = key in ("FL2", "FL11", "HP1")
    ref_xy = colour.XYZ_to_xy(colour.sd_to_XYZ(colour.SDS_ILLUMINANTS[key]))
    np.testing.assert_allclose(cm.xy(binned), ref_xy, atol=5e-4 if spiky else 2e-4)
    ra_ref = colour.colour_rendering_index(colour.SDS_ILLUMINANTS[key])
    assert cm.colour_rendering(binned).ra == pytest.approx(ra_ref, abs=0.3)


def test_luminous_efficacy_of_monochromatic_555nm_is_683():
    from lss.spectral.lines import Line
    from lss.spectral.profiles import Gaussian

    line = Spectrum.of(Line(555.0, 1.0, Gaussian(1e-3))).bin(analysis_grid())
    assert cm.luminous(line) == pytest.approx(683.0 * 1.0, rel=2e-3)  # V(555) = 0.999


def test_rejects_other_grids():
    with pytest.raises(ValueError):
        cm.xy(cie_table("A").__class__(render_grid(), np.ones(len(render_grid())), 0.0, 0.0))
