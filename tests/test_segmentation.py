"""Tests for ABC/XYZ segmentation."""
from src.optimization.segmentation import ABCXYZSegmenter


def test_segmentation_classes(sample_demand):
    seg = ABCXYZSegmenter({"abc_thresholds": [0.80, 0.95], "xyz_thresholds": [0.5, 1.0]})
    result = seg.classify(sample_demand)
    assert set(result["abc_class"].unique()).issubset({"A", "B", "C"})
    assert set(result["xyz_class"].unique()).issubset({"X", "Y", "Z"})
    assert all(result["segment"].str.len() == 2)


def test_policy_assigned(sample_demand):
    seg = ABCXYZSegmenter({})
    result = seg.classify(sample_demand)
    assert result["policy"].notna().all()
