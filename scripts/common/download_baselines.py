#!/usr/bin/env python3
"""Restore the reviewed anonymous baseline snapshots without Git cloning."""
import hashlib
import io
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import subprocess
import zipfile

from normalize_baseline_env import normalize

BASELINES = {
    "SIDReasoner": ("SIDReasoner-05B4", "98710dbbd8362cc828b6ab69d9925132264e283c28fabb457921a23fb0d9ee5b"),
    "SASRec": ("SASRec-1CE8", "580cf4bec269f2b27622071d9104fb86ae6a0be9ad908e3ab3baad5e21836f05"),
}


def restore(root):
    root.mkdir(parents=True, exist_ok=True)
    for name, (anonymous_id, expected) in BASELINES.items():
        target = root / name
        if target.exists() and any(target.iterdir()):
            print(f"{name}: existing files retained; use an empty directory for a fresh download")
            continue
        url = f"https://anonymous.4open.science/api/repo/{anonymous_id}/zip"
        payload = subprocess.check_output([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "2", "--max-time", "120", url,
        ])
        archive = zipfile.ZipFile(io.BytesIO(payload))
        with archive, tempfile.TemporaryDirectory(dir=root) as temporary:
            staging = Path(temporary) / name
            staging.mkdir()
            for entry in archive.infolist():
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in entry.filename:
                    raise ValueError(f"Unsafe archive path: {entry.filename}")
                destination = staging.joinpath(*path.parts)
                if entry.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(entry))
            digest = hashlib.sha256()
            for path in sorted(staging.rglob('*')):
                if path.is_file():
                    digest.update(path.relative_to(staging).as_posix().encode() + b'\0'
                                  + hashlib.sha256(path.read_bytes()).digest())
            if digest.hexdigest() != expected:
                raise ValueError(f"{name}: snapshot differs from the reviewed source; download not installed")
            if name == "SIDReasoner":
                normalize(staging)
            merge_script = staging / 'scripts' / 'merge_fsdp_ckpt_ALL.sh'
            if merge_script.exists():
                merge_script.chmod(0o755)
            if target.exists():
                target.rmdir()
            shutil.move(str(staging), str(target))
        print(f"{name}: downloaded and verified")


if __name__ == '__main__':
    restore(Path(__file__).resolve().parents[2] / 'baselines' / 'repos')
