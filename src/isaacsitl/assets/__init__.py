"""Vendored simulation assets (USD) shipped inside the package.

The airframe specs (`conf/airframe/*.yaml`) name these by bare filename; external
absolute paths keep working for user-supplied craft. Provenance + licensing for every
vendored file is in `NOTICE.md` next to this module.

Isaac-free (pathlib only) so resolving an asset never boots the sim.
"""

from isaacsitl.assets._paths import asset_path

__all__ = ["asset_path"]
