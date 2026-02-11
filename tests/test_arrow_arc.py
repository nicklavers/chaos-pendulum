"""Tests for tapered arc geometry (pure math, no Qt widgets needed).

Tests compute_tapered_arcs() which returns TaperedArc segments encoding
winding numbers as circular tapered arcs. Convention: positive n = CW
(clockwise=True), negative n = CCW (clockwise=False).
"""

import pytest

from fractal.arrow_arc import (
    TaperedArc,
    compute_tapered_arcs,
    ARC_GAP_DEGREES,
    SINGLE_GAP_DEGREES,
)


class TestComputeTaperedArcsZero:
    """n=0 should return empty list (caller draws dotted circle)."""

    def test_returns_empty(self):
        assert compute_tapered_arcs(0) == []


class TestComputeTaperedArcsSinglePositive:
    """n=1: one CW arc spanning 320 degrees (360 - 40)."""

    def test_count(self):
        arcs = compute_tapered_arcs(1)
        assert len(arcs) == 1

    def test_cw_direction(self):
        """Positive n = CW (clockwise=True)."""
        arc = compute_tapered_arcs(1)[0]
        assert arc.clockwise is True

    def test_span(self):
        arc = compute_tapered_arcs(1)[0]
        expected_span = 360.0 - SINGLE_GAP_DEGREES
        assert arc.span_deg == pytest.approx(expected_span)

    def test_start_angle(self):
        """n=1 starts at single_gap/2 = 20 degrees."""
        arc = compute_tapered_arcs(1)[0]
        assert arc.start_deg == pytest.approx(SINGLE_GAP_DEGREES / 2.0)


class TestComputeTaperedArcsMultiple:
    """n=2,3,4: multiple evenly distributed arcs."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
    def test_count_matches_n(self, n: int):
        arcs = compute_tapered_arcs(n)
        assert len(arcs) == n

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_span_per_arc(self, n: int):
        arcs = compute_tapered_arcs(n)
        expected = (360.0 - n * ARC_GAP_DEGREES) / n
        for arc in arcs:
            assert arc.span_deg == pytest.approx(expected)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_all_cw(self, n: int):
        """Positive n = CW (clockwise=True)."""
        arcs = compute_tapered_arcs(n)
        for arc in arcs:
            assert arc.clockwise is True

    def test_arcs_evenly_spaced(self):
        arcs = compute_tapered_arcs(3)
        step = arcs[1].start_deg - arcs[0].start_deg
        for i in range(1, len(arcs)):
            actual_step = arcs[i].start_deg - arcs[i - 1].start_deg
            assert actual_step == pytest.approx(step)

    def test_total_coverage_plus_gaps(self):
        """Total arc spans + gaps should equal 360 degrees."""
        n = 4
        arcs = compute_tapered_arcs(n)
        total_span = sum(a.span_deg for a in arcs)
        total_gap = n * ARC_GAP_DEGREES
        assert total_span + total_gap == pytest.approx(360.0)

    def test_three_arcs_span(self):
        """n=3: each arc spans (360 - 3*4) / 3 = 116 degrees."""
        arcs = compute_tapered_arcs(3)
        expected = (360.0 - 3 * ARC_GAP_DEGREES) / 3
        for arc in arcs:
            assert arc.span_deg == pytest.approx(expected)

    def test_two_arcs_span(self):
        """n=2: each arc spans (360 - 2*4) / 2 = 176 degrees."""
        arcs = compute_tapered_arcs(2)
        expected = (360.0 - 2 * ARC_GAP_DEGREES) / 2
        for arc in arcs:
            assert arc.span_deg == pytest.approx(expected)


class TestComputeTaperedArcsNegative:
    """Negative n: same count but counter-clockwise (CCW)."""

    def test_negative_one_count(self):
        arcs = compute_tapered_arcs(-1)
        assert len(arcs) == 1

    def test_negative_one_ccw(self):
        """Negative n = CCW (clockwise=False)."""
        arc = compute_tapered_arcs(-1)[0]
        assert arc.clockwise is False

    def test_negative_three_count(self):
        arcs = compute_tapered_arcs(-3)
        assert len(arcs) == 3

    def test_negative_three_all_ccw(self):
        arcs = compute_tapered_arcs(-3)
        for arc in arcs:
            assert arc.clockwise is False

    def test_negative_same_spans_as_positive(self):
        """Negative and positive of same |n| should have identical spans."""
        pos = compute_tapered_arcs(3)
        neg = compute_tapered_arcs(-3)
        for p, n in zip(pos, neg):
            assert p.span_deg == pytest.approx(n.span_deg)
            assert p.start_deg == pytest.approx(n.start_deg)


class TestJauntyRotationOffset:
    """All winding numbers should start their first arc at the same angle."""

    def test_n1_start_angle(self):
        """n=1 starts at single_gap/2."""
        arc = compute_tapered_arcs(1)[0]
        expected = SINGLE_GAP_DEGREES / 2.0
        assert arc.start_deg == pytest.approx(expected)

    def test_n2_matches_n1_offset(self):
        """n=2's first arc should start at single_gap/2, not arc_gap/2."""
        arc = compute_tapered_arcs(2)[0]
        expected = ARC_GAP_DEGREES / 2.0 + (SINGLE_GAP_DEGREES - ARC_GAP_DEGREES) / 2.0
        assert arc.start_deg == pytest.approx(expected)
        assert arc.start_deg == pytest.approx(SINGLE_GAP_DEGREES / 2.0)

    def test_n3_matches_n1_offset(self):
        """n=3's first arc should also start at single_gap/2."""
        arc = compute_tapered_arcs(3)[0]
        expected = ARC_GAP_DEGREES / 2.0 + (SINGLE_GAP_DEGREES - ARC_GAP_DEGREES) / 2.0
        assert arc.start_deg == pytest.approx(expected)

    def test_all_first_arcs_same_start(self):
        """All winding numbers 1-5 should have identical first-arc start."""
        starts = [compute_tapered_arcs(n)[0].start_deg for n in range(1, 6)]
        for s in starts:
            assert s == pytest.approx(starts[0])


class TestTaperedArcNamedTuple:
    """TaperedArc is an immutable NamedTuple."""

    def test_fields(self):
        arc = TaperedArc(start_deg=10.0, span_deg=120.0, clockwise=True)
        assert arc.start_deg == 10.0
        assert arc.span_deg == 120.0
        assert arc.clockwise is True

    def test_immutable(self):
        arc = TaperedArc(start_deg=10.0, span_deg=120.0, clockwise=True)
        with pytest.raises(AttributeError):
            arc.start_deg = 20.0  # type: ignore[misc]
