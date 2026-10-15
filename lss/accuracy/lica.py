"""Measured lamps of the LICA-UCM lamp spectra database (design §15), downloaded on first use.

C. Tapia, A. Sánchez de Miguel and J. Zamorano, LICA-UCM lamps spectral database
(UCM eprint 40930, 2015), https://guaix.fis.ucm.es/lamps_spectra. Every file here was
recorded with the same Ocean Optics Jaz spectrograph (serial JAZA2310, as each file's
header states), corrected by LICA for dark signal, non-linearity and the instrument's
spectral response ("flux"), and sampled every 1 nm from 300 to 900 nm. The values are
relative, each spectrum on its own scale.

The site asks to be contacted before its data are used. The repository therefore holds
no LICA file: :func:`fetch` downloads each into the local cache and checks it against
the SHA-256 pinned here, and only the citation, the links and numbers derived from the
data are kept (decision, 2026-10-15).
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from lss import download

PAGE = "https://guaix.fis.ucm.es/lamps_spectra"
BASE_URL = "https://guaix.fis.ucm.es/sites/default/files/lamps_spectra/"
CITATION = ("C. Tapia, A. Sánchez de Miguel and J. Zamorano, LICA-UCM lamps spectral database, "
            "UCM eprint 40930 (2015), " + PAGE)

#: key: (file name, SHA-256 of the file as retrieved 2026-10-15)
FILES = {
    # Narrow-line lamps, for the instrument's response.
    "hg-pencil": ("M_Pen_Ray_Mercury_24382K_LICA_Oriel_cor.txt",
                  "95d78632e63d1e09f095b242b84610c33effb5498a502ee2bdd895b3f38098de"),
    "thorium-argon": ("Th_Thorium_Argon_Photron_cor.txt",
                      "9878f96ff4f933b6537fe2e0a8567f9eff9e6bc5f66f89affaee08999c28eb86"),
    "lps": ("LPS_Low_Pressure_Sodium_1701K_LICA_cor.txt",
            "f28efc1d1777d9b99866fb1c2a2ef56938d85b6ac204e35642ab067c40c26380"),
    # A smooth spectrum, for LICA's response correction.
    "halogen": ("H_Halogen_2911K_LICA_Philips_cor.txt",
                "5d18cefb438474aab0fbd9dadacab3add6d0a9ff0d03331355cb8241bcd4c6fd"),
    # High-pressure sodium street lamps (no electrical data, wattage or arc-tube size).
    "hps-1732k-toledo": ("HPS_High_Pressure_Sodium_1732K_SFH_Toledo_cor.txt",
                         "e51213714b475c05533e96f91ad53faf5cbd98f4903ddeb8794b3d072d63cca4"),
    "hps-1819k-riba-roja-plaza": ("HPS_High_Pressure_Sodium_1819K_Riba_Roja_Plaza_Ajuntament_cor.txt",
                                  "bf22034acb27191c6db0ab9e1611fda9486c4cacf9ee0ec72afc76516db69ee7"),
    "hps-1869k-riba-roja": ("HPS_High_Pressure_Sodium_1869K_Riba_Roja_cor.txt",
                            "15e2fc9c089e8c992e21ccb367027028757cc696f99be07321c7c5c93fd0c02f"),
    "hps-1895k-ager": ("HPS_High_Pressure_Sodium_1895K_Ager_cor.txt",
                       "bee958ee277b5529e657804223dd8da07d7c12034d818a3e65c1051c0694066c"),
    "hps-1920k-alix": ("HPS_High_Pressure_Sodium_1920K_SFH_Alix_cor.txt",
                       "2b04da8ba703a4fe9ca430dcb3894c1b6985f0b6b91a141eab0551e5e1b85fc2"),
    "hps-2005k-fisicas": ("HPS_High_Pressure_Sodium_2005K_F_Fisicas_cor.txt",
                          "6f7ea057b9b212fbd814097f6c3c9af8cc8431aa9159da8e4d329383858c673e"),
    "hps-2038k-riba-roja-plaza": ("HPS_High_Pressure_Sodium_2038K_Riba_Roja_Plaza_Ajuntament_cor.txt",
                                  "d37ed8222999f979c48ee15d63e9a086636f2ab5cff0327b47d7b4d72a3e45dd"),
    "hps-2176k-riba-roja": ("HPS_High_Pressure_Sodium_2176K_Riba_Roja_cor.txt",
                            "c74085462d237fda49e36eb74af440f50ec539cf576a26b259f5d6a4db3e9536"),
    "hps-2255k-paraninfo": ("HPS_High_Pressure_Sodium_2255K_Paraninfo_UCM_cor.txt",
                            "d85e84c5133ca473f2629b933c120e2c2d641c8cf86a914f762af6e6a553010e"),
}
HPS = tuple(k for k in FILES if k.startswith("hps-"))


@dataclass(frozen=True)
class Measured:
    key: str
    header: dict
    wavelengths_nm: np.ndarray
    values: np.ndarray  # relative


def path(key: str) -> Path:
    return download.cache_dir() / "lica" / FILES[key][0]


def available(keys=None) -> bool:
    return all(path(k).exists() for k in (FILES if keys is None else keys))


def fetch(key: str) -> Path:
    name, digest = FILES[key]
    return download.fetch(BASE_URL + name, path(key), digest)


def fetch_all() -> None:
    for key in FILES:
        fetch(key)


@lru_cache(maxsize=None)
def load(key: str) -> Measured:
    header, rows = {}, []
    for line in fetch(key).read_text(encoding="latin-1").splitlines():
        if line.startswith("#"):
            k, _, v = line[1:].partition(":")
            header[k.strip()] = v.strip()
        elif line[:1].isdigit():
            w, v = line.split()[:2]
            rows.append((float(w), float(v)))
    table = np.array(rows)
    if table.size == 0 or np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError(f"{path(key)}: no increasing wavelength table")
    return Measured(key, header, table[:, 0], table[:, 1])
