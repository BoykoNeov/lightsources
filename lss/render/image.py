"""From re-coloured XYZ to a picture (design §3.4: colour only at display time). No Mitsuba.

The picture is white-balanced on the lab's white card (CAT02 to sRGB's D65), as a
camera set on a white card would be, and exposed so the card reads 0.8.
"""

import math
import warnings

import numpy as np

from lss.scene.study import Camera

WHITE_LEVEL = 0.8


def project(camera: Camera, point) -> tuple[float, float]:
    """Pixel (x right, y down) where ``point`` appears; Mitsuba's perspective, fov along x."""
    o, t, up = (np.asarray(v, float) for v in (camera.origin, camera.target, camera.up))
    d = (t - o) / np.linalg.norm(t - o)
    right = np.cross(d, up)
    right /= np.linalg.norm(right)
    u = np.cross(right, d)
    p = np.asarray(point, float) - o
    depth = p @ d
    if depth <= 0:
        raise ValueError("point is behind the camera")
    k = 0.5 * camera.width / math.tan(math.radians(camera.fov_deg) / 2)
    return 0.5 * camera.width + k * (p @ right) / depth, 0.5 * camera.height - k * (p @ u) / depth


def region_mean(xyz: np.ndarray, camera: Camera, centre, half_px: int = 3) -> np.ndarray:
    """Mean XYZ over a small square of pixels around where ``centre`` projects."""
    x, y = project(camera, centre)
    xi, yi = int(round(x)), int(round(y))
    if not (half_px <= xi < camera.width - half_px and half_px <= yi < camera.height - half_px):
        raise ValueError(f"{centre} projects outside the image")
    return xyz[yi - half_px:yi + half_px + 1, xi - half_px:xi + half_px + 1].reshape(-1, 3).mean(axis=0)


def to_srgb(xyz: np.ndarray, white_xyz: np.ndarray) -> np.ndarray:
    """Display sRGB (0-1, encoded), balanced and exposed on ``white_xyz``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour

        white_xy = colour.XYZ_to_xy(white_xyz)
        scaled = xyz * (WHITE_LEVEL / white_xyz[1])
        rgb = colour.XYZ_to_sRGB(scaled, illuminant=white_xy, chromatic_adaptation_transform="CAT02",
                                 apply_cctf_encoding=True)
    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def write_png(path, rgb: np.ndarray) -> None:
    """8-bit PNG, written with the standard library only."""
    import struct
    import zlib

    a = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
