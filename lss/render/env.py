"""Mitsuba setup (design §8.3, §8.4): imported only when rendering, since the renderer
is the optional ``render`` extra.

GPU (``cuda_ad_spectral``) by default. The CPU variant needs an installed LLVM named by
``DRJIT_LIBLLVM_PATH`` and, with LLVM 22 and Dr.Jit 1.5.0, at most 8 worker threads: at
16 it crashes at random on 47-channel kernels (§8.3).
"""

import os

GPU = "cuda_ad_spectral"
CPU = "llvm_ad_spectral"
CPU_THREADS = 8

INSTALL_HINT = ('the renderer needs Mitsuba 3: pip install -e ".[render]" '
                "(see README: the CPU variant also needs DRJIT_LIBLLVM_PATH)")


class RendererUnavailable(RuntimeError):
    pass


def available() -> bool:
    try:
        import mitsuba  # noqa: F401
    except ImportError:
        return False
    return True


def setup(variant: str | None = None) -> str:
    """Select ``variant`` (default: GPU, else CPU) and return it; raise if neither runs."""
    try:
        import drjit as dr
        import mitsuba as mi
    except ImportError as e:
        raise RendererUnavailable(INSTALL_HINT) from e
    wanted = [variant] if variant else [GPU, CPU]
    errors = []
    for v in wanted:
        if v == CPU and not os.environ.get("DRJIT_LIBLLVM_PATH"):
            errors.append(f"{CPU}: DRJIT_LIBLLVM_PATH is not set (an installed LLVM's LLVM-C.dll)")
            continue
        try:
            mi.set_variant(v)
        except (ImportError, AttributeError, RuntimeError) as e:
            errors.append(f"{v}: {e}")
            continue
        if v == CPU:
            dr.set_thread_count(min(CPU_THREADS, os.cpu_count() or CPU_THREADS))
        from lss.render import emitters

        emitters.register()
        return v
    raise RendererUnavailable("no Mitsuba variant runs here: " + "; ".join(errors))
