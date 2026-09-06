from pathlib import Path

from volleycut.eval.validate_truth import validate


def test_actual_unfinished_annotation_packet_cannot_be_frozen():
    root = Path(__file__).resolve().parents[1] / "data/validation_work"
    result = validate(root, verify_images=False)
    assert result["status"] == "not_ready"
    assert sum(c["unlabeled_frames"] for c in result["clips"]) > 0
    assert any("independent rally review not complete" in e for e in result["errors"])
    assert not (root / "frozen_manifest.json").exists()
