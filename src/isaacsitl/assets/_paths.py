"""Resolve vendored USD assets shipped inside the package (Isaac-free, pathlib only)."""

from __future__ import annotations

from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent


def asset_path(name: str) -> Path:
    """Resolve a vendored asset by filename (e.g. ``racer.usd``) to its on-disk path.

    Raises:
        FileNotFoundError: if the asset is not shipped, listing what is.
    """
    path = _ASSETS_DIR / name
    if not path.exists():
        shipped = ", ".join(sorted(p.name for p in _ASSETS_DIR.glob("*.usd")))
        raise FileNotFoundError(f"no vendored asset '{name}' (shipped: {shipped})")
    return path
