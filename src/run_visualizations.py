"""Generate the figures and verify the source data was not touched."""
import hashlib

from src.visualize_results import (
    FIGURES_DIR,
    PAIRWISE_PATH,
    SUMMARY_PATH,
    generate_all_figures,
)


def _hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_visualizations() -> int:
    before = {path: _hash(path) for path in (SUMMARY_PATH, PAIRWISE_PATH)}

    generate_all_figures()

    # Guard against the evaluation source files being overwritten as a side effect.
    for path, digest in before.items():
        if _hash(path) != digest:
            raise RuntimeError(f"Source file was modified during visualization: {path}")

    print(f"Generated figures in {FIGURES_DIR}")
    for path in sorted(FIGURES_DIR.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_visualizations())
