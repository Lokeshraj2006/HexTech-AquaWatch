"""Download the pool datasets from Roboflow Universe (YOLOv11 format) into datasets/raw/<name>/.

Your Roboflow API key is read from the ROBOFLOW_API_KEY environment variable (Roboflow -> Settings ->
API Keys). It is never written to disk or printed. In PowerShell:

    $env:ROBOFLOW_API_KEY = "your_key_here"
    .venv\\Scripts\\python.exe training\\get_datasets.py              # all datasets below
    .venv\\Scripts\\python.exe training\\get_datasets.py university   # just one

Uses only the Python standard library (Roboflow REST export API: ask for an export link, then download
the zip). All three datasets are CC BY 4.0: credit them in the README (see the CREDIT lines printed).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Use IPv4 only: on networks with a broken IPv6 route, api.roboflow.com (which also has an IPv6 address)
# times out instead of answering. IPv4 works everywhere.
_getaddrinfo = socket.getaddrinfo


def _ipv4_only(host, port, family=0, *args, **kwargs):
    return _getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)


socket.getaddrinfo = _ipv4_only

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
FORMAT = "yolov11"

# name -> (workspace, project, version, credit line)
DATASETS = {
    "university": ("university-g3h71", "swimming-and-drowning-detection", 1,
                   "Swimming and Drowning Detection, by University, Roboflow Universe, CC BY 4.0, "
                   "https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection"),
    "poolsafety": ("pool-safety", "drowning-detection-otnme", 3,
                   "drowning detection, by Pool safety, Roboflow Universe, CC BY 4.0, "
                   "https://universe.roboflow.com/pool-safety/drowning-detection-otnme"),
    "yashwanee": ("yashwanee-tetar-jm88u", "drowning-detection-and-prevention-in-swimming-pools", 3,
                  "Drowning Detection And Prevention In Swimming Pools, by yashwanee tetar, Roboflow Universe, "
                  "CC BY 4.0, https://universe.roboflow.com/yashwanee-tetar-jm88u/"
                  "drowning-detection-and-prevention-in-swimming-pools"),
    "treading": ("children-in-buggy", "drowning-i0nae", 1,
                 "drowning (swimming / drowning / treading_water), by children in buggy, Roboflow Universe, CC BY 4.0, "
                 "https://universe.roboflow.com/children-in-buggy/drowning-i0nae"),
}


def _open(url: str, timeout: int, tries: int = 5):
    """urlopen with retries on network errors (timeouts, packet loss, dropped Wi-Fi), with short waits between tries."""
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except urllib.error.HTTPError:
            raise                                     # a real answer from the server: do not retry
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if i == tries - 1:
                raise
            wait = min(3 * (i + 1), 15)
            print(f"   network problem ({getattr(exc, 'reason', exc)}); retrying in {wait} s ...", flush=True)
            time.sleep(wait)


def export_link(workspace: str, project: str, version: int, key: str) -> str:
    """Ask Roboflow to prepare the export and return the zip link (it may take a few tries while it builds)."""
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{FORMAT}?api_key={key}"
    for attempt in range(30):
        try:
            with _open(url, timeout=12, tries=12) as r:      # small request: many quick tries beat one long wait
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            raise SystemExit(f"Roboflow said HTTP {exc.code} for {workspace}/{project} v{version}: {body}\n"
                             "Check the key, or open the dataset page once while signed in (some need 'Fork' first).")
        link = (data.get("export") or {}).get("link")
        if link:
            return link
        if attempt == 0:
            print("   Roboflow is preparing the export...", flush=True)
        time.sleep(5)
    raise SystemExit("export link not ready after 2.5 minutes; try again later")


def download(name: str, key: str) -> None:
    """Download and unzip one dataset into datasets/raw/<name>/ (skipped if data.yaml is already there)."""
    workspace, project, version, credit = DATASETS[name]
    target = RAW / name
    if (target / "data.yaml").exists():
        print(f"== {name}: already in {target}, skipping")
        return
    print(f"== {name}: {workspace}/{project} version {version}", flush=True)
    link = export_link(workspace, project, version, key)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / f"{name}.zip"
        with _open(link, timeout=600) as r, open(zpath, "wb") as fh:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r   {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
        print()
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(target)
    n = sum(1 for _ in target.rglob("*.jpg")) + sum(1 for _ in target.rglob("*.png"))
    print(f"   {n} images in {target}")
    print(f"   CREDIT: {credit}")


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)}")
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set your key first (PowerShell):  $env:ROBOFLOW_API_KEY = \"your_key_here\"")
    RAW.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in names:
        try:
            download(name, key)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f"   FAILED ({getattr(exc, 'reason', exc)}): check the internet connection and run the same "
                  f"command again (finished datasets are skipped).")
            failed.append(name)
    if failed:
        print("\nNot downloaded yet:", ", ".join(failed))
        return 1
    print("\nDone. Next: python training/build_dataset.py (or training/remap_labels.py with a label map per dataset).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
