"""Free system memory and this process's memory, without extra dependencies (Windows and Linux)."""
from __future__ import annotations

import ctypes
import os
import sys


def system_memory() -> dict | None:
    """{"available_mb", "total_mb", "commit_available_mb"} or None if it cannot be read.

    On Windows "commit_available_mb" is what new allocations can still get (RAM + page file):
    when it reaches zero, allocations fail even if Task Manager shows some RAM free.
    """
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return None
            mb = 1024.0 * 1024.0
            return {"available_mb": round(st.ullAvailPhys / mb), "total_mb": round(st.ullTotalPhys / mb),
                    "commit_available_mb": round(st.ullAvailPageFile / mb)}
        info = {}
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0]) / 1024.0
        return {"available_mb": round(info.get("MemAvailable", info.get("MemFree", 0.0))),
                "total_mb": round(info.get("MemTotal", 0.0)),
                "commit_available_mb": round(info.get("MemAvailable", 0.0) + info.get("SwapFree", 0.0))}
    except Exception:
        return None


def process_memory_mb() -> float | None:
    """Private memory of this process in MB (Windows: private bytes, Linux: resident set)."""
    try:
        if sys.platform == "win32":
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                            ("PrivateUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return None
            return round(pmc.PrivateUsage / (1024.0 * 1024.0), 1)
        with open(f"/proc/{os.getpid()}/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        return None
    return None


def is_out_of_memory(exc: BaseException) -> bool:
    """True for Python MemoryError and OpenCV / NumPy allocation failures."""
    if isinstance(exc, MemoryError):
        return True
    text = str(exc)
    return ("Insufficient memory" in text or "OutOfMemoryError" in text
            or "Unable to allocate" in text or "bad allocation" in text)
