from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk up to the repo root (sentinel: .gitmodules next to baselines/)."""
    for c in (start, *start.parents):
        if (c / ".gitmodules").is_file() and (c / "baselines").is_dir():
            return c
    raise FileNotFoundError("repo root (.gitmodules + baselines/) not found")


# Repo root, resolved depth-independently so moving this file never breaks it.
REPO_ROOT = _find_repo_root(Path(__file__).resolve())

# Every experiment config under the (only populated) grpo suite. Globbed rather
# than hard-listed so new configs are picked up automatically.
ALL_CONFIGS = sorted(
    str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "configs" / "grpo").rglob("*.yaml")
)
