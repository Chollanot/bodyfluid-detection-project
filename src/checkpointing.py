"""
checkpointing.py
Crash-safe, resume-everywhere helpers for long Colab runs.

The whole project stores state on Google Drive so that a runtime disconnect
(the #1 Colab failure mode on long jobs) never loses work. Two ideas:

1. A single PROJECT ROOT on Drive holds every artifact: datasets, checkpoints,
   predictions, metrics, figures.
2. `step_done` / `mark_done` implement idempotent stages. Each notebook cell that
   does expensive work is wrapped so that re-running after a crash SKIPS finished
   stages and continues from the first unfinished one.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

DEFAULT_ROOT = "/content/drive/MyDrive/BodyFluidDetection"


def mount_drive():
    """Mount Google Drive inside Colab (no-op elsewhere)."""
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception as e:
        print(f"[checkpointing] Drive not mounted ({e}); using local paths.")


def project_root(dataset: str, root: str = DEFAULT_ROOT) -> Path:
    """Per-dataset root: .../BodyFluidDetection/CX33 or .../Smartphone."""
    p = Path(root) / dataset
    for sub in ("datasets", "checkpoints", "predictions", "metrics", "figures", "state"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def _state_file(root: Path) -> Path:
    return root / "state" / "progress.json"


def _load_state(root: Path) -> dict:
    f = _state_file(root)
    return json.loads(f.read_text()) if f.exists() else {}


def step_done(root: Path, key: str) -> bool:
    return bool(_load_state(root).get(key, {}).get("done"))


def mark_done(root: Path, key: str, **meta):
    st = _load_state(root)
    st[key] = {"done": True, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}
    _state_file(root).write_text(json.dumps(st, indent=2))
    print(f"[checkpointing] stage '{key}' marked done.")


def run_step(root: Path, key: str, fn, force: bool = False):
    """Run fn() only if stage not already done. Returns fn()'s value or None."""
    if step_done(root, key) and not force:
        print(f"[checkpointing] SKIP '{key}' (already complete). "
              f"Pass force=True to redo.")
        return None
    print(f"[checkpointing] RUN '{key}' ...")
    out = fn()
    mark_done(root, key)
    return out
