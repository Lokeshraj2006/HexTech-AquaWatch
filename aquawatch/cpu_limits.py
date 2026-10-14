"""Keep AquaWatch light on a laptop without a GPU: a few CPU threads and below-normal priority.

    import cpu_limits
    cpu_limits.apply()          # at the start of an entry point, and again right after importing torch / ultralytics

AQUAWATCH_THREADS sets the thread count (default 2). AQUAWATCH_PRIORITY=normal keeps the normal process priority.
Why the second call matters: Ultralytics sets torch to NUM_THREADS = min(8, cpu_count - 1) threads every time it
selects the CPU (utils/torch_utils.select_device, on every new predictor), which silently undoes
torch.set_num_threads(2). apply() overrides that NUM_THREADS once Ultralytics is imported. A lowered priority is
inherited by every process started afterwards (run.py started from the dashboard, ingest runs), so the laptop stays
responsive even when an analysis runs.

Nothing heavy is imported here: torch / cv2 / Ultralytics are only touched when the process already imported them.
"""
from __future__ import annotations

import os
import sys

DEFAULT_THREADS = 2
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                   "NUMEXPR_MAX_THREADS", "OPENCV_NUM_THREADS")
_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000      # Windows SetPriorityClass value
_NICE_STEP = 5                                 # POSIX niceness added once


def threads() -> int:
    """The thread budget: AQUAWATCH_THREADS if it is a positive number, else DEFAULT_THREADS."""
    try:
        value = int(os.environ.get("AQUAWATCH_THREADS", DEFAULT_THREADS))
    except ValueError:
        value = DEFAULT_THREADS
    return max(1, value)


def lower_priority() -> bool:
    """Below-normal priority for this process (and the processes it starts). True when it was applied."""
    if os.environ.get("AQUAWATCH_PRIORITY", "").strip().lower() == "normal":
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE       # 64-bit handle: the default int type breaks it
            kernel32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel32.SetPriorityClass.restype = wintypes.BOOL
            return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), _BELOW_NORMAL_PRIORITY_CLASS))
        if os.environ.get("_AQUAWATCH_NICED") != "1":           # nice() adds up: only once per process tree
            os.nice(_NICE_STEP)
            os.environ["_AQUAWATCH_NICED"] = "1"
        return True
    except Exception:                                           # never stop a run because of the priority
        return False


def apply() -> dict:
    """Set the thread budget everywhere it is read and lower the priority. Safe to call any number of times.

    Also keeps the process offline (offline.apply(): no analytics / telemetry / downloads), because every entry point
    already calls this first. Returns what was applied: {"threads", "priority_lowered", "torch", "ultralytics", "cv2"}."""
    import offline
    offline.apply()
    n = threads()
    for name in THREAD_ENV_VARS:                 # read by OpenMP / MKL / OpenCV when they start
        os.environ[name] = str(n)
    done = {"threads": n, "priority_lowered": lower_priority(), "torch": False, "ultralytics": False, "cv2": False}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.set_num_threads(n)
            done["torch"] = True
        except Exception:
            pass
    for module_name in ("ultralytics.utils", "ultralytics.utils.torch_utils"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "NUM_THREADS"):
            module.NUM_THREADS = n               # select_device() reads this module-level name
            done["ultralytics"] = True
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        try:
            cv2.setNumThreads(n)
            done["cv2"] = True
        except Exception:
            pass
    return done
