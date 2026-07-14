from __future__ import annotations

from wombat_transport.transport.convection import _native as _native

for _name in dir(_native):
    if _name in {"__builtins__", "__cached__", "__doc__", "__file__", "__loader__", "__name__", "__package__", "__spec__"}:
        continue
    globals()[_name] = getattr(_native, _name)

__all__ = [name for name in globals() if not name.startswith("__")]
