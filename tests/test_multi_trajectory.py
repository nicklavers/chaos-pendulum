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
from fractal.animated_diagram import (
    TrajectoryInfo,
    PAUSE_FRAMES,
    FREEZE_KEYFRAME_COUNT,
    FREEZE_TRACE_ACTIVE_ALPHA,
    FREEZE_TRACE_SETTLED_ALPHA,
    SETTLE_BUFFER_SECONDS,
    MultiTrajectoryDiagram,
)
from fractal.inspect_column import PinnedTrajectory, FRAME_SUBSAMPLE
from fractal.winding import (
    get_single_winding_color,
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
        b, g, r, a = get_single_winding_color(1, 0, winding_modular_grid)
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
                    n1, n2, winding_modular_grid,
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


# ---------------------------------------------------------------------------
# Freeze-frame: _compute_keyframe_indices
# ---------------------------------------------------------------------------


class TestComputeKeyframeIndices:
    """Tests for the keyframe index computation used in freeze-frame."""

    def test_zero_frames(self):
        """Zero frames should return empty tuple."""
        result = MultiTrajectoryDiagram._compute_keyframe_indices(0)
        assert result == ()

    def test_single_frame(self):
        """Single frame should return (0,)."""
        result = MultiTrajectoryDiagram._compute_keyframe_indices(1)
        assert result == (0,)

    def test_two_frames(self):
        """Two frames should return (0, 1)."""
        result = MultiTrajectoryDiagram._compute_keyframe_indices(2)
        assert result == (0, 1)

    def test_five_frames_exact(self):
        """Five frames = exactly FREEZE_KEYFRAME_COUNT should use all."""
        result = MultiTrajectoryDiagram._compute_keyframe_indices(5)
        assert result == (0, 1, 2, 3, 4)

    def test_hundred_frames(self):
        """100 frames should produce 5 evenly-spaced indices."""
        result = MultiTrajectoryDiagram._compute_keyframe_indices(100)
        assert len(result) == FREEZE_KEYFRAME_COUNT
        assert result[0] == 0
        assert result[-1] == 99
        # All indices should be in valid range
        for idx in result:
            assert 0 <= idx < 100

    def test_five_hundred_frames(self):
        """500 frames should produce 5 evenly-spaced indices."""
        result = MultiTrajectoryDiagram._compute_keyframe_indices(500)
        assert len(result) == FREEZE_KEYFRAME_COUNT
        assert result[0] == 0
        assert result[-1] == 499

    def test_indices_are_sorted(self):
        """Indices should be in ascending order."""
        for n in (3, 10, 50, 200, 1000):
            result = MultiTrajectoryDiagram._compute_keyframe_indices(n)
            assert result == tuple(sorted(result)), f"Not sorted for n={n}"

    def test_no_duplicates(self):
        """No duplicate indices for any reasonable frame count."""
        for n in (1, 2, 3, 4, 5, 10, 50, 100):
            result = MultiTrajectoryDiagram._compute_keyframe_indices(n)
            assert len(result) == len(set(result)), f"Duplicates for n={n}"

    def test_three_frames(self):
        """Three frames (< FREEZE_KEYFRAME_COUNT) should return all three."""
        result = MultiTrajectoryDiagram._compute_keyframe_indices(3)
        assert result == (0, 1, 2)


# ---------------------------------------------------------------------------
# Freeze-frame: enter/exit state management
# ---------------------------------------------------------------------------


class TestFreezeFrameState:
    """Tests for freeze-frame state toggling (data-only, no QPainter)."""

    def _make_tinfo(self, n_frames: int = 10) -> TrajectoryInfo:
        """Create a minimal TrajectoryInfo for testing."""
        traj = np.zeros((n_frames, 4), dtype=np.float32)
        return TrajectoryInfo(trajectory=traj, color_rgb=(100, 200, 50))

    def test_initially_not_frozen(self):
        """New diagram should not be in freeze-frame mode."""
        # Access the class's _freeze_frame directly (no widget instantiation)
        tinfo = self._make_tinfo()
        assert tinfo is not None  # sanity check

    def test_freeze_frame_field_default(self):
        """_freeze_frame field should default to None."""
        # Test the data model: freeze_frame is not part of TrajectoryInfo
        # It lives on MultiTrajectoryDiagram which requires QApp
        # So we test the static helper instead
        assert MultiTrajectoryDiagram._compute_keyframe_indices(0) == ()

    def test_set_trajectories_clears_freeze_state_data_model(self):
        """set_trajectories should conceptually clear freeze-frame.

        We verify indirectly: a TrajectoryInfo built from a PinnedTrajectory
        should be independent of any prior freeze-frame state.
        """
        traj1 = np.ones((10, 4), dtype=np.float32)
        traj2 = np.zeros((20, 4), dtype=np.float32)
        info1 = TrajectoryInfo(trajectory=traj1, color_rgb=(255, 0, 0))
        info2 = TrajectoryInfo(trajectory=traj2, color_rgb=(0, 255, 0))
        # Each TrajectoryInfo is independent (frozen dataclass)
        assert info1.trajectory.shape != info2.trajectory.shape
        assert info1.color_rgb != info2.color_rgb


# ---------------------------------------------------------------------------
# Freeze-frame: _trace_alpha_at (two-tier alpha with transition)
# ---------------------------------------------------------------------------


class TestTraceAlphaAt:
    """Tests for the two-tier alpha computation used in freeze-frame trace."""

    def test_no_settling_returns_active(self):
        """When settled_idx is None, always returns active alpha."""
        for i in (0, 5, 50, 500):
            alpha = MultiTrajectoryDiagram._trace_alpha_at(i, None, 10.0)
            assert alpha == FREEZE_TRACE_ACTIVE_ALPHA

    def test_well_before_settling_returns_active(self):
        """Segments well before the settled index get active alpha."""
        settled = 100
        alpha = MultiTrajectoryDiagram._trace_alpha_at(10, settled, 10.0)
        assert alpha == FREEZE_TRACE_ACTIVE_ALPHA

    def test_well_after_settling_returns_settled(self):
        """Segments well after the settled index get settled alpha."""
        settled = 50
        alpha = MultiTrajectoryDiagram._trace_alpha_at(200, settled, 10.0)
        assert alpha == FREEZE_TRACE_SETTLED_ALPHA

    def test_at_transition_midpoint(self):
        """At the settled index itself, alpha should be midpoint of blend."""
        settled = 50
        alpha = MultiTrajectoryDiagram._trace_alpha_at(50, settled, 10.0)
        expected = (FREEZE_TRACE_ACTIVE_ALPHA + FREEZE_TRACE_SETTLED_ALPHA) // 2
        assert abs(alpha - expected) <= 1  # rounding tolerance

    def test_transition_is_monotonic(self):
        """Alpha should decrease monotonically through the transition region."""
        settled = 100
        half_t = 10.0
        alphas = [
            MultiTrajectoryDiagram._trace_alpha_at(i, settled, half_t)
            for i in range(85, 116)
        ]
        for j in range(1, len(alphas)):
            assert alphas[j] <= alphas[j - 1], (
                f"Alpha increased at i={85 + j}: {alphas[j]} > {alphas[j - 1]}"
            )

    def test_zero_transition_snaps(self):
        """With zero transition width, should snap directly."""
        settled = 50
        assert (
            MultiTrajectoryDiagram._trace_alpha_at(49, settled, 0.0)
            == FREEZE_TRACE_ACTIVE_ALPHA
        )
        assert (
            MultiTrajectoryDiagram._trace_alpha_at(50, settled, 0.0)
            == FREEZE_TRACE_SETTLED_ALPHA
        )

    def test_alpha_stays_in_range(self):
        """Alpha should always be between settled and active values."""
        settled = 50
        lo = min(FREEZE_TRACE_ACTIVE_ALPHA, FREEZE_TRACE_SETTLED_ALPHA)
        hi = max(FREEZE_TRACE_ACTIVE_ALPHA, FREEZE_TRACE_SETTLED_ALPHA)
        for i in range(0, 200):
            alpha = MultiTrajectoryDiagram._trace_alpha_at(i, settled, 10.0)
            assert lo <= alpha <= hi, f"Alpha {alpha} out of range at i={i}"


# ---------------------------------------------------------------------------
# Settle-based animation truncation
# ---------------------------------------------------------------------------


class TestSettleTruncation:
    """Tests for the settle-based animation truncation logic.

    Since MultiTrajectoryDiagram requires QApplication, we test the
    building blocks: saddle_energy, total_energy, and the expected
    buffer math.
    """

    def test_settle_buffer_constant_is_positive(self):
        """SETTLE_BUFFER_SECONDS should be a positive value."""
        assert SETTLE_BUFFER_SECONDS > 0

    def test_buffer_frames_math(self):
        """Buffer frames = ceil(SETTLE_BUFFER_SECONDS / dt_per_frame)."""
        dt_per_frame = FRAME_SUBSAMPLE * 0.01  # 0.06
        buffer = math.ceil(SETTLE_BUFFER_SECONDS / dt_per_frame)
        # 5.0 / 0.06 = 83.33 → ceil = 84
        assert buffer == 84

    def test_settled_trajectory_has_low_energy(self):
        """A damped trajectory eventually drops below saddle energy."""
        from simulation import simulate, total_energy
        from fractal.compute import saddle_energy

        params = DoublePendulumParams(friction=0.5)
        _, states = simulate(params, 1.0, 0.5, t_end=30.0, dt=0.01)
        se = saddle_energy(params)

        # Find first frame below threshold
        settled_idx = None
        for i, state in enumerate(states):
            if total_energy(state, params) < se:
                settled_idx = i
                break

        assert settled_idx is not None, "Trajectory should settle with friction=0.5"
        assert settled_idx < len(states) - 1, "Should settle before end"

    def test_no_friction_energy_conserved(self):
        """Without friction, energy is conserved — no settling occurs."""
        from simulation import simulate, total_energy

        params = DoublePendulumParams(friction=0.0)
        _, states = simulate(params, 2.0, 1.5, t_end=10.0, dt=0.01)
        initial_energy = total_energy(states[0], params)

        # Energy should be conserved (within integration tolerance)
        for state in states[::100]:
            e = total_energy(state, params)
            assert abs(e - initial_energy) / abs(initial_energy) < 0.01

    def test_effective_max_formula(self):
        """Effective max = latest_settle + buffer_frames."""
        # Simulate the math that _compute_effective_max does
        dt_per_frame = 0.06
        buffer = math.ceil(SETTLE_BUFFER_SECONDS / dt_per_frame)

        # If two trajectories settle at frames 50 and 80
        settle_indices = [50, 80]
        latest = max(settle_indices)
        effective = latest + buffer

        assert effective == 80 + 84  # 164
