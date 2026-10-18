"""The flat emitter with an angle band (design §8.5), a Mitsuba emitter written in Python.

``polardisc`` sits on a ``disk`` shape and emits from its front only: radiance ``value``
in the directions whose cosine to the normal μ satisfies ``mu_lo < μ ≤ mu_hi``, none
elsewhere, with a flat unit spectrum (the renderer never sees a lamp spectrum, §8.1).
One band is on at a time; each band's lighting map is rendered by moving the edges.

The edges and the value are kernel inputs, made opaque in ``parameters_changed``: set as
plain values, every new edge compiled the kernel again (§8.5). Built on the spike's arc
emitter (§8.3): what the path tracer needs is written, not ``sample_ray`` (light tracing).
"""

_REGISTERED: set[str] = set()


def register() -> None:
    """Register ``polardisc`` for the current Mitsuba variant (once per variant)."""
    import drjit as dr
    import mitsuba as mi

    if mi.variant() in _REGISTERED:
        return

    class PolarDisc(mi.Emitter):
        def __init__(self, props):
            super().__init__(props)
            self.mu_lo = dr.opaque(mi.Float, props.get("mu_lo", 0.0))
            self.mu_hi = dr.opaque(mi.Float, props.get("mu_hi", 1.0))
            self.value = dr.opaque(mi.Float, props.get("value", 1.0))
            self.m_flags = mi.EmitterFlags.Surface | mi.EmitterFlags.SpatiallyVarying

        def _radiance(self, mu, active):
            return mi.Spectrum(dr.select(active & (mu > self.mu_lo) & (mu <= self.mu_hi) & (mu > 0), self.value, 0.0))

        def eval(self, si, active=True):
            return self._radiance(mi.Frame3f.cos_theta(si.wi), active)

        def sample_direction(self, it, sample, active=True):
            ds = self.get_shape().sample_direction(it, sample, active)
            ok = active & (ds.pdf != 0)
            ds.emitter = mi.EmitterPtr(self)
            return ds, self._radiance(-dr.dot(ds.d, ds.n), ok) / dr.select(ok, ds.pdf, 1.0)

        def pdf_direction(self, it, ds, active=True):
            ok = active & (dr.dot(ds.d, ds.n) < 0)
            return dr.select(ok, self.get_shape().pdf_direction(it, ds, ok), 0.0)

        def eval_direction(self, it, ds, active=True):
            return self._radiance(-dr.dot(ds.d, ds.n), active)

        def sample_ray(self, time, sample1, sample2, sample3, active=True):
            raise NotImplementedError("polardisc: light tracing is not written (the path tracer does not need it)")

        def traverse(self, callback):
            super().traverse(callback)
            for name in ("mu_lo", "mu_hi", "value"):
                callback.put(name, getattr(self, name), mi.ParamFlags.NonDifferentiable)

        def parameters_changed(self, keys):
            dr.make_opaque(self.mu_lo, self.mu_hi, self.value)

        def to_string(self):
            return "PolarDisc"

    mi.register_emitter("polardisc", lambda props: PolarDisc(props))
    _REGISTERED.add(mi.variant())
