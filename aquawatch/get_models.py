"""Get the AquaWatch model files once after cloning (about 57 MB): the three pool detectors (OpenVINO exports) and the
pose model. They are too large for the repository, so they are attached to the project's GitHub release.

    python aquawatch/get_models.py                      # download, check, unpack into the project folder
    python aquawatch/get_models.py --zip models.zip     # from a copy you already have (no internet needed)
    python aquawatch/get_models.py --check              # only show which model files are here
    python aquawatch/get_models.py --with-chat-model    # also the CLIP file for the chat (about 338 MB, optional)

Standard library only, so it runs right after pip install. Every file is checked with SHA-256 before it is used.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent           # the project folder (this file is in aquawatch/)

MODELS_URL = ("https://github.com/Lokeshraj2006/HexTech-AquaWatch/releases/download/models-v1/"
              "aquawatch-models-v1.zip")
MODELS_SHA256 = "2eb7307936ccce23913c90cfd2f8e1aababa5c160362c10874d4932dc04caba1"
# OpenAI CLIP ViT-B/32, from the same address the OpenAI "clip" package uses; its SHA-256 is part of the address
CLIP_URL = ("https://openaipublic.azureedge.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")
CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

MODELS = [("Standard detector (preset pool)", "models/quant/yolo11n_openvino_fp32/yolo11n_openvino_model"),
          ("Wide CCTV detector (preset pool_hd)", "models/quant/yolo11s_openvino_fp32/yolo11s_openvino_model"),
          ("Swimmer posture model (preset pool_trained)", "models/quant/best_openvino_fp32/best_openvino_model"),
          ("Pose model (arm and head signals)", "yolo11n-pose.pt")]
CHAT_MODEL = ("Chat model (CLIP, optional)", "weights/clip/ViT-B-32.pt")


class SetupError(Exception):
    """A problem the user can fix (no internet, a damaged file): shown as one plain message, no traceback."""


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MB pieces."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, label: str) -> None:
    """Download url to dest (through dest.part, so a broken download never looks finished)."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label} ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done, shown = 0, -5
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                pct = int(done * 100 / total) if total else -1
                if total and pct >= shown + 5:
                    shown = pct - pct % 5
                    print(f"  {done / 1e6:6.1f} of {total / 1e6:.1f} MB ({pct} %)", flush=True)
    except Exception as exc:                            # no internet, proxy, 404 ...: one clear message
        part.unlink(missing_ok=True)
        raise SetupError(f"Could not download {label} from {url}\n  ({type(exc).__name__}: {exc})\n"
                         "  Check the internet connection, or download the file in a browser and run\n"
                         "  python aquawatch/get_models.py --zip <the downloaded file>") from None
    part.replace(dest)


def check_sha256(path: Path, expected: str, label: str) -> None:
    """Stop when the file is not exactly the published one (a damaged or a different file)."""
    got = sha256(path)
    if got != expected:
        raise SetupError(f"{label} is damaged or not the expected file (SHA-256 {got[:12]}..., expected "
                         f"{expected[:12]}...). Delete {path} and try again.")
    print(f"  SHA-256 checked: {got[:12]}...", flush=True)


def safe_extract(zip_path: Path, root: Path) -> list[str]:
    """Unpack the zip into root; refuse any entry that would land outside it (an absolute path or '..')."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise SetupError(f"{zip_path} is not a zip file.") from None
    with zf:
        names = zf.namelist()
        for name in names:
            parts = PurePosixPath(name.replace("\\", "/")).parts
            if name.startswith(("/", "\\")) or ".." in parts or (parts and ":" in parts[0]):
                raise SetupError(f"Refusing to unpack {zip_path}: the entry '{name}' points outside the project folder.")
        zf.extractall(root)
    return names


def present(rel: str) -> bool:
    return (ROOT / rel).exists()


def report() -> bool:
    """Print which model files are here; True when all four pool models are."""
    for label, rel in MODELS + [CHAT_MODEL]:
        print(f"  {'OK     ' if present(rel) else 'missing'}  {label}: {rel}")
    return all(present(rel) for _, rel in MODELS)


def install_models(zip_arg: str | None) -> None:
    """Get the models zip (download or a local copy), check it, unpack it."""
    if zip_arg:
        zip_path = Path(zip_arg).expanduser()
        if not zip_path.is_file():
            raise SetupError(f"No such file: {zip_path}")
    else:
        zip_path = ROOT / "models" / "aquawatch-models-v1.zip"
        if not zip_path.is_file():
            download(MODELS_URL, zip_path, "the AquaWatch models (about 57 MB)")
    check_sha256(zip_path, MODELS_SHA256, "The models zip")
    names = safe_extract(zip_path, ROOT)
    print(f"  unpacked {len(names)} files into {ROOT}", flush=True)
    if not zip_arg:
        zip_path.unlink(missing_ok=True)                # the unpacked files are what the app uses


def install_chat_model() -> None:
    """Download the optional CLIP file the chat uses (question routing and picture search)."""
    dest = ROOT / CHAT_MODEL[1]
    if dest.is_file():
        print("The chat model is already here.")
        return
    download(CLIP_URL, dest, "the chat model (CLIP ViT-B/32, about 338 MB)")
    try:
        check_sha256(dest, CLIP_SHA256, "The chat model")
    except SetupError:
        dest.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="get_models.py", description="Get the AquaWatch model files (once, after cloning).")
    p.add_argument("--zip", metavar="FILE", help="use a copy of aquawatch-models-v1.zip you already have")
    p.add_argument("--check", action="store_true", help="only show which model files are here")
    p.add_argument("--with-chat-model", action="store_true",
                   help="also download the optional chat model (CLIP, about 338 MB)")
    p.add_argument("--force", action="store_true", help="unpack the models again even if they are here")
    args = p.parse_args(argv)

    if args.check:
        return 0 if report() else 1
    try:
        if all(present(rel) for _, rel in MODELS) and not (args.force or args.zip):
            print("All pool model files are already here.")
        else:
            install_models(args.zip)
        if args.with_chat_model:
            install_chat_model()
    except SetupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Model files:")
    ok = report()
    print("Ready: start the dashboard with start_aquawatch.bat or: streamlit run app.py" if ok
          else "Some model files are still missing (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
