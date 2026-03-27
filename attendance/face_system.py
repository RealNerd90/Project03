from __future__ import annotations

from typing import Optional

from attendance_system import AttendanceSystem

_system: Optional[AttendanceSystem] = None


def get_system() -> AttendanceSystem:
    """
    Return a singleton AttendanceSystem instance shared across Django views.

    This keeps the face database loaded only once and reuses the same model.
    """
    global _system
    if _system is None:
        _system = AttendanceSystem(database_path="media")
    return _system

