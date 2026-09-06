"""Resident-memory measurement, on whichever platform is running.

Four near-identical copies of the Windows PSAPI block had accumulated across the
evaluation harness and one script. They type-checked on Windows and failed on
Linux -- `ctypes.windll` does not exist there -- which is how CI broke while the
local run stayed green.

Every platform-specific branch is guarded by a `sys.platform` comparison rather
than a runtime `os.name` check. mypy narrows on those comparisons, so when it
type-checks for Linux it never looks inside the Windows branch, and vice versa.
That is what makes one file correct on both platforms without per-line ignores,
which are themselves platform-specific and get flagged as unused on the other
side.

Every function returns None rather than raising when the platform cannot answer:
these numbers annotate experiment manifests, and a missing measurement must not
fail a run that otherwise succeeded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _windows_working_set() -> tuple[int, int] | None:
    """(current, peak) working-set bytes from PSAPI, or None off Windows."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        if not get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            return None
    except (AttributeError, OSError, ValueError):
        return None
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def peak_process_rss_bytes() -> int | None:
    """Peak resident set size for this process, or None if unavailable."""
    if sys.platform == "win32":
        counters = _windows_working_set()
        return counters[1] if counters else None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, ValueError):
        return None
    # getrusage reports kilobytes on Linux and bytes on macOS.
    return value if sys.platform == "darwin" else value * 1024


def current_process_rss_bytes() -> int | None:
    """Current resident set size, falling back to the peak where unavailable."""
    if sys.platform == "win32":
        counters = _windows_working_set()
        return counters[0] if counters else None
    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError):
            return None
    return peak_process_rss_bytes()
