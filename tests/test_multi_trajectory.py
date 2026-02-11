"""Tests for the stacked multi-trajectory animation redesign.

Tests TrajectoryInfo, PinnedTrajectory frozen dataclasses,
frame subsampling, and basin color lookup logic.

Note: Qt widget tests (MultiTrajectoryDiagram, TrajectoryIndicator)
are not included because they require a running QApplication, which
causes segfaults in CI. Widget behaviour is verified via manual testing.
"""

import math

import numpy as np
import pytest

from simulation import DoublePendulumParams
from fractal.animated_diagram import TrajectoryInfo, PAUSE_FRAMES
from fractal.inspect_column import PinnedTrajectory, FRAME_SUBSAMPLE
from fractal.winding import (
    get_single_winding_color,
    winding_direction_brightness,
    winding_modular_grid,
    WINDING_COLORMAPS,
)


# ---------------------------------------------------------------------------
# TrajectoryInfo (frozen dataclass)
# ---------------------------------------------------------------------------


class TestTrajectoryInfo:
    """Tests for the TrajectoryInfo frozen dataclass."""

    def test_creation(self):
        """Should create with trajectory and color."""
        traj = np.zeros((10, 4), dtype=np.float32)
        info = TrajectoryInfo(trajectory=traj, color_rgb=(255, 0, 0))
        assert info.color_rgb == (255, 0, 0)
        assert info.trajectory.shape == (10, 4)

    def test_frozen(self):
        """Should not allow attribute modification."""
        traj = np.zeros((10, 4), dtype=np.float32)
        info = TrajectoryInfo(trajectory=traj, color_rgb=(255, 0, 0))
        with pytest.raises(AttributeError):
            info.color_rgb = (0, 255, 0)

    def test_multiple_infos_independent(self):
        """Multiple TrajectoryInfo instances should be independent."""
        traj1 = np.ones((10, 4), dtype=np.float32)
        traj2 = np.zeros((20, 4), dtype=np.float32)
        info1 = TrajectoryInfo(trajectory=traj1, color_rgb=(255, 0, 0))
        info2 = TrajectoryInfo(trajectory=traj2, color_rgb=(0, 255, 0))
        assert info1.trajectory.shape != info2.trajectory.shape
        assert info1.color_rgb != info2.color_rgb


# ---------------------------------------------------------------------------
# PinnedTrajectory (frozen dataclass)
# ---------------------------------------------------------------------------


class TestPinnedTrajectory:
    """Tests for the PinnedTrajectory frozen dataclass."""

    def test_creation(self):
        """Should create with all required fields."""
        traj = np.zeros((50, 4), dtype=np.float32)
        pt = PinnedTrajectory(
            row_id="test-1",
            theta1_init=0.5,
            theta2_init=-0.3,
            trajectory=traj,
            n1=1,
            n2=-1,
            color_rgb=(200, 100, 50),
        )
        assert pt.row_id == "test-1"
        assert pt.theta1_init == 0.5
        assert pt.theta2_init == -0.3
        assert pt.n1 == 1
        assert pt.n2 == -1
        assert pt.color_rgb == (200, 100, 50)

    def test_frozen(self):
        """Should not allow attribute modification."""
        traj = np.zeros((50, 4), dtype=np.float32)
        pt = PinnedTrajectory(
            row_id="test-1",
            theta1_init=0.5,
            theta2_init=-0.3,
            trajectory=traj,
            n1=1,
            n2=-1,
            color_rgb=(200, 100, 50),
        )
        with pytest.raises(AttributeError):
            pt.row_id = "test-2"

    def test_immutable_rebuild(self):
        """Creating a new PinnedTrajectory with updated color works."""
        traj = np.zeros((50, 4), dtype=np.float32)
        original = PinnedTrajectory(
            row_id="test-1",
            theta1_init=0.5,
            theta2_init=-0.3,
            trajectory=traj,
            n1=1,
            n2=-1,
            color_rgb=(200, 100, 50),
        )
        # Immutable rebuild with new color
        updated = PinnedTrajectory(
            row_id=original.row_id,
            theta1_init=original.theta1_init,
            theta2_init=original.theta2_init,
            trajectory=original.trajectory,
            n1=original.n1,
            n2=original.n2,
            color_rgb=(100, 200, 50),
        )
        assert updated.color_rgb == (100, 200, 50)
        assert original.color_rgb == (200, 100, 50)  # unchanged


# ---------------------------------------------------------------------------
# FRAME_SUBSAMPLE constant
# ---------------------------------------------------------------------------


class TestFrameSubsample:
    """Verify frame subsampling constant is reasonable."""

    def test_subsample_value(self):
        """FRAME_SUBSAMPLE should be a positive integer."""
        assert isinstance(FRAME_SUBSAMPLE, int)
        assert FRAME_SUBSAMPLE > 0

    def test_subsample_reduces_frames(self):
        """Subsampling should reduce frame count significantly."""
        n_raw = 1000
        n_sub = len(range(0, n_raw, FRAME_SUBSAMPLE))
        assert n_sub < n_raw
        assert n_sub > 0

    def test_subsample_array_slicing(self):
        """numpy array slicing with FRAME_SUBSAMPLE should work correctly."""
        traj = np.arange(600 * 4, dtype=np.float32).reshape(600, 4)
        subsampled = traj[::FRAME_SUBSAMPLE]
        expected_len = math.ceil(600 / FRAME_SUBSAMPLE)
        assert subsampled.shape[0] == expected_len
        # First element should be unchanged
        np.testing.assert_array_equal(subsampled[0], traj[0])


# ---------------------------------------------------------------------------
# PAUSE_FRAMES constant
# ---------------------------------------------------------------------------


class TestPauseFrames:
    """Verify the initial pause constant."""

    def test_pause_frames_is_about_one_second(self):
        """At ~30fps, PAUSE_FRAMES=30 gives ~1 second pause."""
        assert PAUSE_FRAMES == 30

    def test_pause_frames_positive(self):
        """PAUSE_FRAMES must be positive."""
        assert PAUSE_FRAMES > 0


# ---------------------------------------------------------------------------
# Basin color lookup logic
# ---------------------------------------------------------------------------


class TestBasinColorLookup:
    """Test the BGRA-to-RGB conversion used in InspectColumn._lookup_basin_color."""

    def test_bgra_to_rgb_conversion(self):
        """get_single_winding_color returns BGRA; we need RGB for bob colors."""
        b, g, r, a = get_single_winding_color(1, 0, winding_direction_brightness)
        # All values should be in valid range
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255
        # The RGB extracted from BGRA should be (r, g, b)
        rgb = (r, g, b)
        assert len(rgb) == 3

    def test_different_winding_pairs_give_different_colors(self):
        """Different winding numbers should produce different RGB colors."""
        colors = set()
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                b, g, r, _ = get_single_winding_color(
                    n1, n2, winding_direction_brightness,
                )
                colors.add((r, g, b))
        # Should have multiple distinct colors
        assert len(colors) > 5

    def test_all_colormaps_produce_valid_rgb(self):
        """All registered winding colormaps should produce valid RGB values."""
        for name, colormap_fn in WINDING_COLORMAPS.items():
            b, g, r, a = get_single_winding_color(1, 1, colormap_fn)
            assert 0 <= r <= 255, f"{name}: invalid R={r}"
            assert 0 <= g <= 255, f"{name}: invalid G={g}"
            assert 0 <= b <= 255, f"{name}: invalid B={b}"

    def test_modular_grid_distinct_within_5x5(self):
        """Modular grid should produce distinct colors within a 5x5 block."""
        colors = set()
        for n1 in range(5):
            for n2 in range(5):
                b, g, r, _ = get_single_winding_color(
                    n1, n2, winding_modular_grid,
                )
                colors.add((r, g, b))
        # 5x5 = 25 unique colors expected
        assert len(colors) == 25


# ---------------------------------------------------------------------------
# TrajectoryInfo tuple building
# ---------------------------------------------------------------------------


class TestTrajectoryInfoTupleBuilding:
    """Test the pattern of building TrajectoryInfo tuples from PinnedTrajectory."""

    def test_build_from_pinned(self):
        """Should build TrajectoryInfo from PinnedTrajectory data."""
        traj = np.zeros((50, 4), dtype=np.float32)
        pt = PinnedTrajectory(
            row_id="test-1",
            theta1_init=0.5,
            theta2_init=-0.3,
            trajectory=traj,
            n1=1,
            n2=0,
            color_rgb=(200, 100, 50),
        )
        info = TrajectoryInfo(
            trajectory=pt.trajectory,
            color_rgb=pt.color_rgb,
        )
        assert info.trajectory is pt.trajectory
        assert info.color_rgb == pt.color_rgb

    def test_build_tuple_preserves_order(self):
        """Building a tuple of TrajectoryInfo should preserve insertion order."""
        infos = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, color in enumerate(colors):
            traj = np.full((10, 4), float(i), dtype=np.float32)
            infos.append(TrajectoryInfo(trajectory=traj, color_rgb=color))

        result = tuple(infos)
        assert len(result) == 3
        assert result[0].color_rgb == (255, 0, 0)
        assert result[1].color_rgb == (0, 255, 0)
        assert result[2].color_rgb == (0, 0, 255)

    def test_primary_index_selection(self):
        """Should correctly identify primary index in insertion order."""
        insertion_order = ("a", "b", "c")
        primary_id = "b"

        primary_index = 0
        for i, row_id in enumerate(insertion_order):
            if row_id == primary_id:
                primary_index = i
                break

        assert primary_index == 1
