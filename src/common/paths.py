"""Shared path utilities.

`find_repo_root` resolves the repository root by walking up from a starting
path until it finds a sentinel that only exists at the root (``.gitmodules``
plus the ``baselines`` directory). This is depth-independent: moving a script
or test deeper/shallower in the tree never breaks root resolution (unlike
``Path(__file__).parents[N]`` counting).
"""

from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """Return the repo root by searching upward for the root sentinels.

    Args:
        start: A file or directory inside the repo (typically ``__file__``).
            Defaults to this module's location.

    Raises:
        FileNotFoundError: if no ancestor contains the sentinels.
    """
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".gitmodules").is_file() and (candidate / "baselines").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate repo root (sentinels .gitmodules + baselines/) above {here}"
    )
