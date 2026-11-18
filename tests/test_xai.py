"""Tests for detection matching and occlusion bookkeeping."""
import numpy as np

from explainability.run_xai import grid_boxes, hide, match_truth, spatial_stats, target_score


def test_match_truth_one_to_one_and_class():
    truths = [(0, (0, 0, 10, 10))]
    preds = [{"class_id": 0, "class_name": "D00", "confidence": .9, "xyxy": [0, 0, 10, 10]},
             {"class_id": 0, "class_name": "D00", "confidence": .8, "xyxy": [0, 0, 10, 10]},
             {"class_id": 1, "class_name": "D10", "confidence": .7, "xyxy": [0, 0, 10, 10]}]
    rows, fn = match_truth(preds, truths)
    assert [r["status"] for r in rows] == ["TP", "FP", "FP"]
    assert fn == 0


def test_target_score_ignores_order_and_wrong_class():
    target = {"class_id": 0, "confidence": .6, "xyxy": [0, 0, 10, 10]}
    preds = [{"class_id": 1, "confidence": .99, "xyxy": [0, 0, 10, 10]},
             {"class_id": 0, "confidence": .5, "xyxy": [0, 0, 10, 10]}]
    assert target_score(preds, target) == .5
    assert target_score(list(reversed(preds)), target) == .5
    assert target_score([], target) == 0


def test_reference_score_can_exceed_target_confidence():
    target = {"class_id": 0, "confidence": .6, "xyxy": [0, 0, 10, 10]}
    predictions = [target, {"class_id": 0, "confidence": .8,
                            "xyxy": [0, 0, 10, 10]}]
    assert target_score(predictions, target) == .8


def test_regions_and_stats():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    fill = np.ones_like(image)
    boxes = grid_boxes(8, 8, 2)
    assert len(boxes) == 4
    changed = hide(image, fill, boxes[0])
    assert changed[:4, :4].sum() == 48
    assert image.sum() == 0
    stats = spatial_stats(np.array([[1., -1.], [0., 0.]]),
                          {"xyxy": [0, 0, 4, 4]}, 8, 8)
    assert stats["inside_mean"] == 1
    assert stats["negative_cells"] == 1


def test_small_target_box_receives_fractional_attribution():
    values = np.zeros((2, 2))
    values[0, 0] = 1
    stats = spatial_stats(values, {"xyxy": [0, 0, 2, 2]}, 8, 8)
    assert stats["inside_mean"] == 1
    assert stats["inside_fraction"] == .25
    assert 0 < stats["outside_mean"] < 1
