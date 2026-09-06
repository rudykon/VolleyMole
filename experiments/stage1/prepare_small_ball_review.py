"""Native annotation navigation with a 25px color-crop radius cap.

This does not run a model, read predictions, or write labels. The radius cap
excludes large shirt/box/sign contours that dominate the original navigation
heuristic. Missing proposals never establish invisibility; every point still
requires individual source-image review. The original helper stays unchanged
so prior annotation evidence hashes remain valid.
"""
import importlib.util
from pathlib import Path


def main():
    path = Path(__file__).resolve().parents[1] / "stage0/prepare_visual_review.py"
    spec = importlib.util.spec_from_file_location("native_review_original", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    candidates = original.candidates

    def small_candidates(frame, previous):
        choices, gray = candidates(frame, previous)
        return [c for c in choices if c["radius"] <= 25], gray

    original.candidates = small_candidates
    original.main()


if __name__ == "__main__":
    main()
