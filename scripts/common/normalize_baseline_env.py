#!/usr/bin/env python3
"""Give upstream cluster environment scripts a portable name before patching."""
from pathlib import Path
import os
import re
import sys


def source_files(root):
    """Avoid rewriting downloaded data, checkpoints, or installed environments."""
    excluded = {"data", "checkpoints", "pretrained", "results", "outputs", "logs",
                "wandb", "runs", "venv", "env", "__pycache__", "node_modules"}
    suffixes = {".py", ".sh", ".md", ".rst", ".job", ".sbatch", ".toml", ".yaml", ".yml"}
    for directory, folders, files in os.walk(root):
        folders[:] = [name for name in folders
                      if not name.startswith(".") and name not in excluded]
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink() and (path.suffix in suffixes or name == "Makefile"):
                yield path


def normalize(root):
    root = Path(root)
    for directory in (root / "scripts", root / "final_scripts"):
        for source in sorted(directory.glob("*_env.sh")):
            if source.name == "cluster_env.sh":
                continue
            setup = source.read_text()
            if "PYTHON_CMD=" not in setup or "TORCHRUN_CMD=" not in setup:
                continue
            target = source.with_name("cluster_env.sh")
            if target.exists():
                raise RuntimeError(f"Cannot rename {source}: {target} already exists")
            old_name = source.stem.removesuffix("_env")
            pattern = re.compile(re.escape(old_name), re.IGNORECASE)
            for path in source_files(root):
                try:
                    content = path.read_text()
                except UnicodeError:
                    continue
                updated = re.sub(re.escape(old_name.capitalize()) + r" cluster", "Slurm cluster", content)
                updated = pattern.sub(
                    lambda match: "cluster" if match[0].islower() else "the cluster", updated
                )
                if updated != content:
                    path.write_text(updated)
            source.rename(target)


if __name__ == "__main__":
    normalize(sys.argv[1])
