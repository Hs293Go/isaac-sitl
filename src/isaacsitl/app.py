"""Boot the Isaac Kit app — call `launch()` BEFORE importing `isaacsitl.sim`.

The Kit lifecycle rule: `isaaclab` modules (and anything importing them, i.e.
`isaacsitl.sim`) may only be imported after the SimulationApp exists. Scripts
with their own CLI should keep the standard `AppLauncher.add_app_launcher_args`
pattern instead (see `examples/nav.py`); `launch()` is the library-consumer
shortcut.
"""

from __future__ import annotations

from typing import Any


def launch(headless: bool = True, device: str = "cpu", **kwargs: Any):
    """Boot Kit; returns the SimulationApp (keep it alive; `.close()` when done)."""
    from isaaclab.app import AppLauncher

    return AppLauncher({"headless": headless, "device": device, **kwargs}).app
