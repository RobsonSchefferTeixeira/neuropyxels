"""
Single source of truth for colormap names available in the app, plus
a resolver that returns a pyqtgraph ColorMap for ANY name matplotlib
can resolve.

Why the resolver exists
-----------------------
pyqtgraph's ``colormap.get(name)`` only resolves a small set of names
natively (its own map files under ``pyqtgraph/colors/maps``) plus a
handful of matplotlib aliases. It does NOT fall back to matplotlib's
full catalog. So a shared list built from pyqtgraph's own registry
collapses to ~8 entries, and names like ``Greys`` raise
``FileNotFoundError`` even though matplotlib happily renders them.

This module bridges that gap: ``get_pg_colormap(name)`` tries
pyqtgraph first, and on failure samples the matplotlib colormap
into a ``pg.ColorMap``. Result: every name matplotlib resolves works
in every figure, `_r` variants included.

The name list
-------------
``COLORMAP_NAMES`` is the matplotlib catalog (with ColorCET
registered under ``cet_*`` if colorcet is installed), filtered to
names matplotlib itself can resolve. That's the same set the
spectrogram has always used, so nothing the user could pick before
disappears.

ColorCET
--------
Imported at module load so its maps register in matplotlib under
``cet_*``. If colorcet isn't installed, those names simply aren't
in the list.
"""

from __future__ import annotations

import warnings

# Importing colorcet registers its maps in matplotlib as cet_<name>.
try:
    import colorcet  # noqa: F401  (side effect: registers in matplotlib)
    _HAS_COLORCET = True
except ImportError:
    _HAS_COLORCET = False


# ---- Candidate list, in display order ----
# Grouped by matplotlib family. Every entry is verified at import
# time against matplotlib's registry; entries that don't resolve on
# this install are silently dropped. Ordering below is the order the
# user sees in the dropdown.
_CANDIDATES: list[str] = [
    # Perceptual heads first -- what most users actually want.
    "viridis", "plasma", "inferno", "magma", "cividis",
    # Sequential
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu",
    "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
    "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink",
    "spring", "summer", "autumn", "winter", "cool", "Wistia",
    "hot", "afmhot", "gist_heat", "copper",
    # Diverging
    "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu",
    "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic",
    # Cyclic
    "twilight", "twilight_shifted", "hsv",
    # Qualitative
    "Pastel1", "Pastel2", "Paired", "Accent",
    "Dark2", "Set1", "Set2", "Set3",
    "tab10", "tab20", "tab20b", "tab20c",
    # Miscellaneous
    "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern",
    "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg",
    "gist_rainbow", "rainbow", "jet", "turbo", "nipy_spectral",
]


# ---- ColorCET shortlist ----
# Only names that actually exist. Verified below against matplotlib;
# any that don't resolve on this install drop out.
_COLORCET_CANDIDATES = [
    "cet_fire", "cet_fire_r",
    "cet_rainbow", "cet_rainbow_r",
    "cet_glasbey", "cet_glasbey_bw", "cet_glasbey_hv",
    "cet_gray", "cet_gray_r",
    "cet_blues", "cet_greens", "cet_reds",
    "cet_bmy", "cet_bmy_r",
    "cet_kbc", "cet_kbc_r",
    "cet_diverging_bwr_40_95_c42",
    "cet_diverging_bwr_55_98_c37",
]


def _matplotlib_can_resolve(name: str) -> bool:
    try:
        from matplotlib import colormaps as _cm
        _cm.get_cmap(name)
        return True
    except Exception:
        return False


def _build_colormap_names() -> list[str]:
    """Filter the candidate list against matplotlib's registry. Names
    that don't resolve here won't resolve anywhere else downstream
    either (the pyqtgraph figures now go through matplotlib via
    get_pg_colormap), so this is the only validity check that
    matters."""
    names: list[str] = []
    for name in _CANDIDATES:
        if name in names:
            continue
        if _matplotlib_can_resolve(name):
            names.append(name)

    if _HAS_COLORCET:
        for name in _COLORCET_CANDIDATES:
            if name in names:
                continue
            if _matplotlib_can_resolve(name):
                names.append(name)

    if not names:
        warnings.warn(
            "No colormaps resolved; falling back to ['viridis', 'plasma', 'inferno'].",
            RuntimeWarning,
            stacklevel=2,
        )
        names = ["viridis", "plasma", "inferno"]

    return names


COLORMAP_NAMES: list[str] = _build_colormap_names()


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

# Cache of constructed ColorMaps, keyed by resolved name (with any
# _r suffix). Building one samples matplotlib 256 times, which is
# cheap, but no reason to do it on every redraw.
_PG_COLORMAP_CACHE: dict[str, "object"] = {}


def _matplotlib_colormap_to_pg(name: str):
    """Sample a matplotlib colormap into a pyqtgraph ColorMap.

    pyqtgraph's ColorMap takes (positions, colors) where positions is
    a 1-D array of length N in [0, 1] and colors is an (N, 4) uint8
    RGBA array. We sample 256 points, which matches pyqtgraph's own
    internal resolution and is visually indistinguishable from
    matplotlib's continuous interpolation at any reasonable display
    size.
    """
    from matplotlib import colormaps as _cm
    import numpy as np
    import pyqtgraph as pg

    cmap = _cm.get_cmap(name)
    n = 256
    positions = np.linspace(0.0, 1.0, n)
    rgba = cmap(positions)                       # (n, 4) float in [0, 1]
    rgba_255 = (rgba * 255.0 + 0.5).astype(np.uint8)
    return pg.ColorMap(positions, rgba_255)


def get_pg_colormap(name: str):
    """Resolve a colormap name to a pyqtgraph ColorMap.

    Order of attempts:
      1. ``pg.colormap.get(name)`` (native pyqtgraph map -- matches
         pyqtgraph's own figures exactly).
      2. ``_matplotlib_colormap_to_pg(name)`` (matplotlib fallback --
         covers every name matplotlib can resolve, including all the
         ``_r`` variants).

    Raises ValueError if neither works, with a message that names the
    attempted map so a typo in a saved session is diagnosable.
    """
    import pyqtgraph as pg

    cached = _PG_COLORMAP_CACHE.get(name)
    if cached is not None:
        return cached

    try:
        cmap = pg.colormap.get(name)
    except Exception:
        cmap = None

    if cmap is None:
        try:
            cmap = _matplotlib_colormap_to_pg(name)
        except Exception as exc:
            raise ValueError(
                f"Colormap {name!r} is not resolvable via pyqtgraph "
                f"or matplotlib: {exc}"
            ) from exc

    _PG_COLORMAP_CACHE[name] = cmap
    return cmap


def resolve_name(name: str, reverse: bool = False) -> str:
    """Append the ``_r`` suffix when reverse is requested, and fall
    back to the first entry of COLORMAP_NAMES if ``name`` isn't in the
    current environment's list.

    Idempotent: if ``name`` already ends with ``_r`` (either because
    the source list contains a reversed variant like ``cet_kbc_r``, or
    because a saved session stored one), the suffix is not appended a
    second time. Without this guard, `resolve_name("cet_kbc_r",
    reverse=True)` would produce `cet_kbc_r_r`, which matplotlib
    rejects."""
    if name not in COLORMAP_NAMES:
        name = COLORMAP_NAMES[0] if COLORMAP_NAMES else "viridis"

    if not reverse:
        return name
    if name.endswith("_r"):
        return name
    return name + "_r"


def get_matplotlib_colormap(name: str):
    """Resolve a name to a matplotlib Colormap. Used by the
    spectrogram, which does its own image rendering with matplotlib
    and does not go through pyqtgraph at all."""
    from matplotlib import colormaps as _cm
    return _cm.get_cmap(name)