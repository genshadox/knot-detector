"""
Tests for filter_boxes_ioa_merge (board.py).
"""

import pytest

from board import filter_boxes_ioa_merge


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def test_isolated_box_unchanged():
    """A box with no overlapping neighbor should pass through unchanged."""

    box = (0.0, 0.0, 10.0, 10.0)
    conf = 0.9

    result = filter_boxes_ioa_merge([(box, conf)], ioa_threshold=0.3)

    assert len(result) == 1
    result_box, result_conf = result[0]

    assert result_box == pytest.approx(box)
    assert result_conf == pytest.approx(conf)


def test_simple_pair_merges():
    """Two boxes with high mutual IoA (one mostly inside the other) merge
    into their union box, with the max confidence of the two."""

    box_a = (0.0, 0.0, 10.0, 10.0)  # big box
    box_b = (2.0, 2.0, 8.0, 8.0)  # fully inside box_a
    conf_a, conf_b = 0.5, 0.9

    result = filter_boxes_ioa_merge(
        [(box_a, conf_a), (box_b, conf_b)], ioa_threshold=0.3
    )

    assert len(result) == 1
    merged_box, merged_conf = result[0]

    # union of a fully-containing pair == the bigger box's extent
    assert merged_box == pytest.approx(box_a)
    assert merged_conf == pytest.approx(max(conf_a, conf_b))


def test_three_box_chain_merges():
    """A-B overlap, B-C overlap, but A-C do NOT directly overlap enough
    to merge on their own. All three should still end up as one merged
    box, where B is acting as a bridge."""

    # laid out left to right, each overlapping only its immediate neighbor
    box_a = (0.0, 0.0, 10.0, 10.0)
    box_b = (8.0, 0.0, 18.0, 10.0)  # overlaps box_a (x: 8-10) and box_c (x: 16-18)
    box_c = (16.0, 0.0, 26.0, 10.0)  # does NOT overlap box_a

    boxes_xyxy_conf = [(box_a, 0.6), (box_b, 0.7), (box_c, 0.8)]

    # make sure A and C really don't overlap
    assert box_a[2] < box_c[0], "test setup invalid: box_a and box_c should not overlap"

    result = filter_boxes_ioa_merge(boxes_xyxy_conf, ioa_threshold=0.15)

    assert len(result) == 1, "A-B-C should merge into a single box via the B bridge"

    merged_box, merged_conf = result[0]

    assert merged_box == pytest.approx((0.0, 0.0, 26.0, 10.0))  # union of all three
    assert merged_conf == pytest.approx(0.8)


def test_two_separate_clusters_stay_separate():
    """Two clusters of overlapping boxes, far apart from each other,
    should merge internally but never merge across clusters."""

    # cluster 1: two overlapping boxes near the origin
    cluster1_a = (0.0, 0.0, 10.0, 10.0)
    cluster1_b = (2.0, 2.0, 8.0, 8.0)

    # cluster 2: two overlapping boxes far away, no overlap with cluster 1
    cluster2_a = (1000.0, 1000.0, 1010.0, 1010.0)
    cluster2_b = (1002.0, 1002.0, 1008.0, 1008.0)

    boxes_xyxy_conf = [
        (cluster1_a, 0.5),
        (cluster1_b, 0.6),
        (cluster2_a, 0.7),
        (cluster2_b, 0.8),
    ]

    result = filter_boxes_ioa_merge(boxes_xyxy_conf, ioa_threshold=0.3)
 
    assert len(result) == 2, "expected exactly 2 merged boxes, one per cluster"
 
    # one merged box should be near the origin, the other near (1000,1000) --
    # not mixed together into a single box spanning both
    origin_cluster_count = 0
    far_cluster_count = 0
 
    for merged_box, _merged_conf in result:
        center_x, _center_y = box_center(merged_box)
 
        if center_x < 100:
            origin_cluster_count += 1
        elif center_x > 900:
            far_cluster_count += 1
 
    assert origin_cluster_count == 1
    assert far_cluster_count == 1


def test_empty_input_returns_empty():
    """No boxes in, no boxes out -- shouldn't error on an empty board."""

    result = filter_boxes_ioa_merge([], ioa_threshold=0.3)

    assert result == []
