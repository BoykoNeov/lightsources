"""Spectral lines: centre, integrated strength, profile, and wavelength medium."""

from dataclasses import dataclass, field
from typing import Literal

from lss.spectral.profiles import Profile
from lss.spectral.units import vacuum_to_air


@dataclass(frozen=True)
class Line:
    """One line. ``power`` is the integral over the profile, in the spectrum's units × nm.

    ``medium`` says whether ``centre_nm`` is an air or a vacuum wavelength.
    NIST lists air wavelengths above 200 nm and vacuum below; analysis grids
    and the CIE tables are in air, so binning always uses :attr:`air_centre`.
    """

    centre_nm: float
    power: float
    profile: Profile
    medium: Literal["air", "vacuum"] = "air"
    label: str = field(default="", compare=False)

    def __post_init__(self):
        if self.medium not in ("air", "vacuum"):
            raise ValueError("medium must be 'air' or 'vacuum'")

    @property
    def air_centre(self) -> float:
        if self.medium == "air":
            return float(self.centre_nm)
        return float(vacuum_to_air(self.centre_nm))
