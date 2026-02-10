"""Tests for fractal/bivariate.py: torus colormaps, pipeline, legend."""

import math

import numpy as np
import pytest

from fractal.bivariate import (
    torus_hue_lightness,
    torus_rgb_sinusoid,
    torus_rgb_sinusoid_aligned,
    torus_rgb_aligned_yb,
    torus_rgb_aligned_gm,
    torus_rgb_aligned_ybgm,
    torus_rgb_aligned_6color,
    torus_warm_cool,
    torus_diagonal_hue,
    bivariate_to_argb,
    build_torus_legend,
    TORUS_COLORMAPS,
)


class TestTorusColormapOutputShape:
    """All torus colormaps should return (N, 4) uint8 BGRA arrays."""

    @pytest.mark.parametrize("fn_name", list(TORUS_COLORMAPS.keys()))
    def test_output_shape_and_dtype(self, fn_name):
        fn = TORUS_COLORMAPS[fn_name]
        n = 100
        t1 = np.random.uniform(0, 2 * math.pi, n).astype(np.float32)
        t2 = np.random.uniform(0, 2 * math.pi, n).astype(np.float32)
        result = fn(t1, t2)
        assert result.shape == (n, 4)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("fn_name", list(TORUS_COLORMAPS.keys()))
    def test_single_element(self, fn_name):
        fn = TORUS_COLORMAPS[fn_name]
        t1 = np.array([1.0], dtype=np.float32)
        t2 = np.array([2.0], dtype=np.float32)
        result = fn(t1, t2)
        assert result.shape == (1, 4)

    @pytest.mark.parametrize("fn_name", list(TORUS_COLORMAPS.keys()))
    def test_alpha_channel_opaque(self, fn_name):
        fn = TORUS_COLORMAPS[fn_name]
        t1 = np.linspace(0, 2 * math.pi, 50, dtype=np.float32)
        t2 = np.linspace(0, 2 * math.pi, 50, dtype=np.float32)
        result = fn(t1, t2)
        assert np.all(result[:, 3] == 255)


class TestTorusPeriodicWrapping:
    """Torus colormaps must produce identical colors at theta and theta + 2pi."""

    @pytest.mark.parametrize("fn_name", list(TORUS_COLORMAPS.keys()))
    def test_theta1_periodicity(self, fn_name):
        fn = TORUS_COLORMAPS[fn_name]
        t1 = np.array([0.5, 1.0, 2.0, 4.5], dtype=np.float32)
        t2 = np.array([0.3, 0.7, 1.5, 3.0], dtype=np.float32)
        a = fn(t1, t2)
        b = fn(t1 + np.float32(2 * math.pi), t2)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("fn_name", list(TORUS_COLORMAPS.keys()))
    def test_theta2_periodicity(self, fn_name):
        fn = TORUS_COLORMAPS[fn_name]
        t1 = np.array([0.5, 1.0, 2.0, 4.5], dtype=np.float32)
        t2 = np.array([0.3, 0.7, 1.5, 3.0], dtype=np.float32)
        a = fn(t1, t2)
        b = fn(t1, t2 + np.float32(2 * math.pi))
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("fn_name", list(TORUS_COLORMAPS.keys()))
    def test_both_periodic(self, fn_name):
        fn = TORUS_COLORMAPS[fn_name]
        t1 = np.array([1.0, 3.0], dtype=np.float32)
        t2 = np.array([2.0, 5.0], dtype=np.float32)
        a = fn(t1, t2)
        b = fn(
            t1 + np.float32(2 * math.pi),
            t2 + np.float32(2 * math.pi),
        )
        np.testing.assert_array_equal(a, b)


class TestTorusColormapDiscrimination:
    """Different (theta1, theta2) pairs should produce different colors."""

    @pytest.mark.parametrize("fn_name", list(TORUS_COLORMAPS.keys()))
    def test_different_inputs_different_colors(self, fn_name):
        fn = TORUS_COLORMAPS[fn_name]
        # Use (0, 0) vs (pi/2, pi) — avoids symmetry in diagonal hue
        t1_a = np.array([0.0], dtype=np.float32)
        t2_a = np.array([0.0], dtype=np.float32)
        t1_b = np.array([math.pi / 2], dtype=np.float32)
        t2_b = np.array([math.pi], dtype=np.float32)
        a = fn(t1_a, t2_a)
        b = fn(t1_b, t2_b)
        # At least one RGB channel should differ
        assert not np.array_equal(a[:, :3], b[:, :3])

    @pytest.mark.parametrize("fn_name", list(TORUS_COLORMAPS.keys()))
    def test_theta2_variation_changes_color(self, fn_name):
        """Varying only theta2 while holding theta1 fixed should change color."""
        fn = TORUS_COLORMAPS[fn_name]
        t1 = np.array([1.0, 1.0], dtype=np.float32)
        t2 = np.array([0.0, math.pi], dtype=np.float32)
        result = fn(t1, t2)
        assert not np.array_equal(result[0, :3], result[1, :3])


class TestTorusColormapValues:
    """Spot-check specific colormap values."""

    def test_rgb_sinusoid_at_zero(self):
        t1 = np.array([0.0], dtype=np.float32)
        t2 = np.array([0.0], dtype=np.float32)
        result = torus_rgb_sinusoid(t1, t2)
        # R = 128 + 127*sin(0) = 128
        assert result[0, 2] == 128  # R channel (BGRA index 2)

    def test_diagonal_hue_equal_angles_bright(self):
        """When theta1 == theta2, cos(0) = 1, so value should be high."""
        t1 = np.array([1.0], dtype=np.float32)
        t2 = np.array([1.0], dtype=np.float32)
        result = torus_diagonal_hue(t1, t2)
        # V = 0.55 + 0.45 * cos(0) = 1.0, so colors should be bright
        # At least one channel should be > 200
        assert np.max(result[0, :3]) > 200


class TestAlignedLandmarks:
    """Verify that aligned variants place landmarks at the correct positions."""

    LANDMARKS = [
        # (theta1, theta2, expected_R, expected_G, expected_B, label)
        (math.pi, math.pi, 255, 1, 1, "red at (pi,pi)"),
        (0.0, 0.0, 1, 1, 1, "black at (0,0)"),
        (0.0, math.pi, 1, 255, 255, "cyan at (0,pi)"),
        (math.pi, 0.0, 255, 255, 255, "white at (pi,0)"),
    ]

    ALL_ALIGNED_FNS = [
        torus_rgb_sinusoid_aligned,
        torus_rgb_aligned_yb,
        torus_rgb_aligned_gm,
        torus_rgb_aligned_ybgm,
        torus_rgb_aligned_6color,
    ]

    @pytest.mark.parametrize(
        "fn", ALL_ALIGNED_FNS,
        ids=["aligned", "yb", "gm", "ybgm", "6color"],
    )
    @pytest.mark.parametrize(
        "t1,t2,exp_r,exp_g,exp_b,label", LANDMARKS,
        ids=[lm[5] for lm in LANDMARKS],
    )
    def test_landmark_colors(self, fn, t1, t2, exp_r, exp_g, exp_b, label):
        theta1 = np.array([t1], dtype=np.float32)
        theta2 = np.array([t2], dtype=np.float32)
        result = fn(theta1, theta2)
        # BGRA order: [B, G, R, A]
        assert result[0, 2] == exp_r, f"{label}: R expected {exp_r}, got {result[0, 2]}"
        assert result[0, 1] == exp_g, f"{label}: G expected {exp_g}, got {result[0, 1]}"
        assert result[0, 0] == exp_b, f"{label}: B expected {exp_b}, got {result[0, 0]}"

    def test_original_differs_from_aligned(self):
        """Original and aligned should differ at the same input points."""
        t1 = np.array([math.pi, 0.0], dtype=np.float32)
        t2 = np.array([math.pi, 0.0], dtype=np.float32)
        original = torus_rgb_sinusoid(t1, t2)
        aligned = torus_rgb_sinusoid_aligned(t1, t2)
        assert not np.array_equal(original[:, :3], aligned[:, :3])

    def test_variants_differ_at_intermediate_points(self):
        """All aligned variants should differ from each other at non-landmark points."""
        t1 = np.array([math.pi / 4, math.pi / 3], dtype=np.float32)
        t2 = np.array([math.pi / 4, math.pi / 6], dtype=np.float32)
        results = [fn(t1, t2) for fn in self.ALL_ALIGNED_FNS]
        # Each pair should differ in at least one RGB channel
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert not np.array_equal(
                    results[i][:, :3], results[j][:, :3]
                ), f"Variants {i} and {j} are identical at intermediate points"


class TestHalfPiLandmarks:
    """Verify that half-pi variants place correct colors at diagonal half-pi points."""

    def _eval(self, fn, t1_val, t2_val):
        """Evaluate colormap at a single point, return (R, G, B)."""
        t1 = np.array([t1_val], dtype=np.float32)
        t2 = np.array([t2_val], dtype=np.float32)
        result = fn(t1, t2)
        return result[0, 2], result[0, 1], result[0, 0]  # R, G, B from BGRA

    # Yellow/Blue variant: diagonal half-pi landmarks
    # Note: min channel is 1, not 0, because 128 + 127*(-1) = 1
    YB_LANDMARKS = [
        # (theta1, theta2, exp_R, exp_G, exp_B, label)
        (math.pi / 2, math.pi / 2, 255, 255, 1, "yellow at (pi/2,pi/2)"),
        (3 * math.pi / 2, 3 * math.pi / 2, 255, 255, 1, "yellow at (3pi/2,3pi/2)"),
        (math.pi / 2, 3 * math.pi / 2, 1, 1, 255, "blue at (pi/2,3pi/2)"),
        (3 * math.pi / 2, math.pi / 2, 1, 1, 255, "blue at (3pi/2,pi/2)"),
    ]

    @pytest.mark.parametrize(
        "t1,t2,exp_r,exp_g,exp_b,label", YB_LANDMARKS,
        ids=[lm[5] for lm in YB_LANDMARKS],
    )
    def test_yb_half_pi_landmarks(self, t1, t2, exp_r, exp_g, exp_b, label):
        r, g, b = self._eval(torus_rgb_aligned_yb, t1, t2)
        assert r == exp_r, f"{label}: R expected {exp_r}, got {r}"
        assert g == exp_g, f"{label}: G expected {exp_g}, got {g}"
        assert b == exp_b, f"{label}: B expected {exp_b}, got {b}"

    # Green/Magenta variant: diagonal half-pi landmarks
    # Note: min channel is 1, not 0, because 128 + 127*(-1) = 1
    GM_LANDMARKS = [
        (math.pi / 2, math.pi / 2, 1, 255, 1, "green at (pi/2,pi/2)"),
        (3 * math.pi / 2, 3 * math.pi / 2, 1, 255, 1, "green at (3pi/2,3pi/2)"),
        (math.pi / 2, 3 * math.pi / 2, 255, 1, 255, "magenta at (pi/2,3pi/2)"),
        (3 * math.pi / 2, math.pi / 2, 255, 1, 255, "magenta at (3pi/2,pi/2)"),
    ]

    @pytest.mark.parametrize(
        "t1,t2,exp_r,exp_g,exp_b,label", GM_LANDMARKS,
        ids=[lm[5] for lm in GM_LANDMARKS],
    )
    def test_gm_half_pi_landmarks(self, t1, t2, exp_r, exp_g, exp_b, label):
        r, g, b = self._eval(torus_rgb_aligned_gm, t1, t2)
        assert r == exp_r, f"{label}: R expected {exp_r}, got {r}"
        assert g == exp_g, f"{label}: G expected {exp_g}, got {g}"
        assert b == exp_b, f"{label}: B expected {exp_b}, got {b}"

    # YBGM variant: four diagonal half-pi landmarks, each a distinct color
    # Note: min channel is 1, not 0, because 128 + 127*(-1) = 1
    YBGM_LANDMARKS = [
        (math.pi / 2, math.pi / 2, 255, 255, 1, "yellow at (pi/2,pi/2)"),
        (math.pi / 2, 3 * math.pi / 2, 1, 1, 255, "blue at (pi/2,3pi/2)"),
        (3 * math.pi / 2, math.pi / 2, 255, 1, 255, "magenta at (3pi/2,pi/2)"),
        (3 * math.pi / 2, 3 * math.pi / 2, 1, 255, 1, "green at (3pi/2,3pi/2)"),
    ]

    @pytest.mark.parametrize(
        "t1,t2,exp_r,exp_g,exp_b,label", YBGM_LANDMARKS,
        ids=[lm[5] for lm in YBGM_LANDMARKS],
    )
    def test_ybgm_half_pi_landmarks(self, t1, t2, exp_r, exp_g, exp_b, label):
        r, g, b = self._eval(torus_rgb_aligned_ybgm, t1, t2)
        assert r == exp_r, f"{label}: R expected {exp_r}, got {r}"
        assert g == exp_g, f"{label}: G expected {exp_g}, got {g}"
        assert b == exp_b, f"{label}: B expected {exp_b}, got {b}"

    def test_ybgm_all_four_colors_distinct(self):
        """YBGM should place 4 distinct colors at the 4 diagonal half-pi points."""
        colors = set()
        for t1_val, t2_val, *_ in self.YBGM_LANDMARKS:
            r, g, b = self._eval(torus_rgb_aligned_ybgm, t1_val, t2_val)
            colors.add((r, g, b))
        assert len(colors) == 4, f"Expected 4 distinct colors, got {len(colors)}"

    # 6-Color variant: all 16 half-pi grid points
    # Note: min channel is 1, not 0, because 128 + 127*(-1) = 1
    SIXCOLOR_LANDMARKS = [
        # Row t1=0
        (0.0, 0.0, 1, 1, 1, "black at (0,0)"),
        (0.0, math.pi / 2, 1, 1, 255, "blue at (0,pi/2)"),
        (0.0, math.pi, 1, 255, 255, "cyan at (0,pi)"),
        (0.0, 3 * math.pi / 2, 1, 255, 1, "green at (0,3pi/2)"),
        # Row t1=pi/2
        (math.pi / 2, 0.0, 1, 255, 1, "green at (pi/2,0)"),
        (math.pi / 2, math.pi / 2, 255, 255, 1, "yellow at (pi/2,pi/2)"),
        (math.pi / 2, math.pi, 255, 1, 255, "magenta at (pi/2,pi)"),
        (math.pi / 2, 3 * math.pi / 2, 1, 1, 255, "blue at (pi/2,3pi/2)"),
        # Row t1=pi
        (math.pi, 0.0, 255, 255, 255, "white at (pi,0)"),
        (math.pi, math.pi / 2, 255, 255, 1, "yellow at (pi,pi/2)"),
        (math.pi, math.pi, 255, 1, 1, "red at (pi,pi)"),
        (math.pi, 3 * math.pi / 2, 255, 1, 255, "magenta at (pi,3pi/2)"),
        # Row t1=3pi/2
        (3 * math.pi / 2, 0.0, 255, 1, 255, "magenta at (3pi/2,0)"),
        (3 * math.pi / 2, math.pi / 2, 1, 1, 255, "blue at (3pi/2,pi/2)"),
        (3 * math.pi / 2, math.pi, 1, 255, 1, "green at (3pi/2,pi)"),
        (3 * math.pi / 2, 3 * math.pi / 2, 255, 255, 1, "yellow at (3pi/2,3pi/2)"),
    ]

    @pytest.mark.parametrize(
        "t1,t2,exp_r,exp_g,exp_b,label", SIXCOLOR_LANDMARKS,
        ids=[lm[5] for lm in SIXCOLOR_LANDMARKS],
    )
    def test_6color_all_landmarks(self, t1, t2, exp_r, exp_g, exp_b, label):
        r, g, b = self._eval(torus_rgb_aligned_6color, t1, t2)
        assert r == exp_r, f"{label}: R expected {exp_r}, got {r}"
        assert g == exp_g, f"{label}: G expected {exp_g}, got {g}"
        assert b == exp_b, f"{label}: B expected {exp_b}, got {b}"


class TestBivariateToArgb:
    """Test the bivariate_to_argb pipeline function."""

    def test_output_shape(self):
        t1 = np.zeros(16, dtype=np.float32)
        t2 = np.zeros(16, dtype=np.float32)
        result = bivariate_to_argb(t1, t2, torus_hue_lightness, 4)
        assert result.shape == (4, 4, 4)
        assert result.dtype == np.uint8

    def test_larger_grid(self):
        n = 64 * 64
        t1 = np.random.uniform(0, 6, n).astype(np.float32)
        t2 = np.random.uniform(0, 6, n).astype(np.float32)
        result = bivariate_to_argb(t1, t2, torus_rgb_sinusoid, 64)
        assert result.shape == (64, 64, 4)

    def test_wrapping(self):
        t1_a = np.array([1.0], dtype=np.float32)
        t2_a = np.array([1.0], dtype=np.float32)
        t1_b = np.array([1.0 + 2 * math.pi], dtype=np.float32)
        t2_b = np.array([1.0 + 2 * math.pi], dtype=np.float32)
        a = bivariate_to_argb(t1_a, t2_a, torus_hue_lightness, 1)
        b = bivariate_to_argb(t1_b, t2_b, torus_hue_lightness, 1)
        np.testing.assert_array_equal(a, b)

    def test_uniform_input_uniform_output(self):
        """All identical (theta1, theta2) should produce identical pixels."""
        n = 9
        t1 = np.full(n, 1.5, dtype=np.float32)
        t2 = np.full(n, 2.5, dtype=np.float32)
        result = bivariate_to_argb(t1, t2, torus_warm_cool, 3)
        first = result[0, 0]
        for i in range(3):
            for j in range(3):
                np.testing.assert_array_equal(result[i, j], first)


class TestBuildTorusLegend:
    """Test the 2D legend builder."""

    def test_output_shape(self):
        legend = build_torus_legend(torus_hue_lightness, size=32)
        assert legend.shape == (32, 32, 4)
        assert legend.dtype == np.uint8

    def test_default_size(self):
        legend = build_torus_legend(torus_rgb_sinusoid)
        assert legend.shape == (64, 64, 4)

    def test_all_opaque(self):
        legend = build_torus_legend(torus_diagonal_hue, size=16)
        assert np.all(legend[:, :, 3] == 255)


class TestTorusColormapPerformance:
    """Verify bivariate coloring meets the 10ms budget at 256x256."""

    @pytest.mark.parametrize("fn_name", list(TORUS_COLORMAPS.keys()))
    def test_performance_budget(self, fn_name):
        import time
        fn = TORUS_COLORMAPS[fn_name]
        n = 256 * 256
        t1 = np.random.uniform(0, 2 * math.pi, n).astype(np.float32)
        t2 = np.random.uniform(0, 2 * math.pi, n).astype(np.float32)

        # Warmup
        fn(t1, t2)

        start = time.perf_counter()
        for _ in range(10):
            fn(t1, t2)
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 0.010, f"{fn_name} took {elapsed:.4f}s (budget: 0.010s)"


class TestTorusColormapRegistry:
    """Test the TORUS_COLORMAPS registry."""

    def test_at_least_three_colormaps(self):
        assert len(TORUS_COLORMAPS) >= 3

    def test_nine_colormaps(self):
        assert len(TORUS_COLORMAPS) == 9

    def test_all_callable(self):
        for name, fn in TORUS_COLORMAPS.items():
            assert callable(fn), f"{name} is not callable"

    def test_all_names_present(self):
        expected = {
            "Hue-Lightness Torus",
            "RGB Sinusoid Torus",
            "RGB Sinusoid Aligned",
            "RGB Aligned + Yellow/Blue",
            "RGB Aligned + Green/Magenta",
            "RGB Aligned + YBGM",
            "RGB Aligned + 6-Color",
            "Warm-Cool Torus",
            "Diagonal Hue Torus",
        }
        assert set(TORUS_COLORMAPS.keys()) == expected
