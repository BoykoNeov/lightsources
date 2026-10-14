"""Physical parameters with units and provenance (design §11, "no hidden fudge factors").

Every parameter is either a physical quantity with a unit and a source, or is
explicitly tagged ``empirical`` with a reason. Climbing a model ladder means
replacing empirical parameters with mechanisms; the accuracy report counts them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    value: float
    unit: str
    source: str = ""
    empirical: bool = False
    reason: str = ""
    bounds: tuple[float, float] | None = None

    def __post_init__(self):
        if self.empirical and not self.reason:
            raise ValueError("an empirical parameter needs a reason")
        if not self.empirical and not self.source:
            raise ValueError("a physical parameter needs a source")
        if self.bounds is not None and not (self.bounds[0] <= self.value <= self.bounds[1]):
            raise ValueError(f"value {self.value} outside bounds {self.bounds}")

    @classmethod
    def from_dict(cls, d: dict) -> "Parameter":
        bounds = d.get("bounds")
        return cls(
            value=float(d["value"]),
            unit=str(d["unit"]),
            source=str(d.get("source", "")),
            empirical=bool(d.get("empirical", False)),
            reason=str(d.get("reason", "")),
            bounds=tuple(bounds) if bounds is not None else None,
        )

    def as_dict(self) -> dict:
        d = {"value": self.value, "unit": self.unit}
        if self.source:
            d["source"] = self.source
        if self.empirical:
            d["empirical"] = True
            d["reason"] = self.reason
        return d
