"""Tests for fractal/coloring.py: LUT correctness, color mapping, QImage."""

import math

import numpy as np
import pytest

from fractal.coloring import (
    build_hue_lut, angle_to_argb, interpolate_angle, numpy_to_qimage,
    DEFAULT_LUT_SIZE,
)


class TestBuildHueLut:
    """Test HSV hue LUT construction."""

    def test_default_size(self):
        lut = build_hue_lut()
        assert lut.shape == (DEFAULT_LUT_SIZE, 4)
        assert lut.dtype == np.uint8

    def test_custom_size(self):
        lut = build_hue_lut(size=256)
        assert lut.shape == (256, 4)

    def test_alpha_channel_opaque(self):
        lut = build_hue_lut()
        assert np.all(lut[:, 3] == 255)

    def test_hue_0_is_red(self):
        """Hue 0 should be pure red: R=255, G=0, B=0."""
        lut = build_hue_lut(size=360)
        # BGRA order: [B, G, R, A]
        assert lut[0, 2] == 255  # R
        assert lut[0, 1] == 0    # G
        assert lut[0, 0] == 0    # B

    def test_hue_120_is_green(self):
        """Hue 120 should be pure green."""
        lut = build_hue_lut(size=360)
        assert lut[120, 1] == 255  # G
        assert lut[120, 2] == 0    # R
        assert lut[120, 0] == 0    # B

    def test_hue_240_is_blue(self):
        """Hue 240 should be pure blue."""
        lut = build_hue_lut(size=360)
        assert lut[240, 0] == 255  # B
        assert lut[240, 1] == 0    # G
        assert lut[240, 2] == 0    # R


class TestAngleToArgb:
    """Test angle -> ARGB pixel mapping."""

    def test_output_shape(self):
        lut = build_hue_lut()
        theta2 = np.zeros(16, dtype=np.float32)
        argb = angle_to_argb(theta2, lut, resolution=4)
        assert argb.shape == (4, 4, 4)
        assert argb.dtype == np.uint8

    def test_uniform_input_uniform_output(self):
        """All identical theta2 values should produce identical pixels."""
        lut = build_hue_lut()
        theta2 = np.full(9, 1.0, dtype=np.float32)
        argb = angle_to_argb(theta2, lut, resolution=3)

        first_pixel = argb[0, 0]
        for i in range(3):
            for j in range(3):
                np.testing.assert_array_equal(argb[i, j], first_pixel)

    def test_wrapping_symmetry(self):
        """theta2 and theta2 + 2*pi should produce the same color."""
        lut = build_hue_lut()
        theta2_a = np.array([1.0], dtype=np.float32)
        theta2_b = np.array([1.0 + 2 * math.pi], dtype=np.float32)

        argb_a = angle_to_argb(theta2_a, lut, resolution=1)
        argb_b = angle_to_argb(theta2_b, lut, resolution=1)

        np.testing.assert_array_equal(argb_a, argb_b)


class TestInterpolateAngle:
    """Test time-index interpolation of angle snapshots."""

    def test_integer_index(self):
        """Integer index should return exact sample."""
        snapshots = np.arange(40, dtype=np.float32).reshape(4, 10)
        result = interpolate_angle(snapshots, 5.0)
        np.testing.assert_array_equal(result, snapshots[:, 5])

    def test_fractional_index(self):
        """Fractional index should interpolate linearly."""
        snapshots = np.array([[0.0, 10.0]], dtype=np.float32)
        result = interpolate_angle(snapshots, 0.5)
        assert abs(result[0] - 5.0) < 1e-5

    def test_boundary_clamp(self):
        """Out-of-range indices should clamp to valid range."""
        snapshots = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result_low = interpolate_angle(snapshots, -1.0)
        result_high = interpolate_angle(snapshots, 100.0)
        assert abs(result_low[0] - 1.0) < 1e-5
        assert abs(result_high[0] - 3.0) < 1e-5

    def test_output_shape(self):
        n, samples = 100, 96
        snapshots = np.random.randn(n, samples).astype(np.float32)
        result = interpolate_angle(snapshots, 50.3)
        assert result.shape == (n,)
        assert result.dtype == np.float32


class TestNumpyToQimage:
    """Test QImage construction from numpy array."""

    def test_creates_valid_qimage(self):
        argb = np.zeros((8, 8, 4), dtype=np.uint8)
        argb[:, :, 3] = 255  # alpha
        image = numpy_to_qimage(argb)
        assert image.width() == 8
        assert image.height() == 8
        assert not image.isNull()

    def test_gc_safety(self):
        """QImage should hold reference to numpy array."""
        argb = np.zeros((4, 4, 4), dtype=np.uint8)
        image = numpy_to_qimage(argb)
        assert hasattr(image, '_numpy_ref')
        assert image._numpy_ref is not None


class TestAngleIndexColoring:
    """Test that angle index selection produces correct coloring output."""

    def test_different_angles_produce_different_colors(self):
        """Theta1 and theta2 with different values should produce different images."""
        lut = build_hue_lut()
        n = 4  # 2x2 grid
        n_samples = 10

        snapshots = np.zeros((n, 2, n_samples), dtype=np.float32)
        snapshots[:, 0, :] = 0.5   # theta1 = 0.5
        snapshots[:, 1, :] = 2.5   # theta2 = 2.5

        theta1_colors = angle_to_argb(
            interpolate_angle(snapshots[:, 0, :], 0.0), lut, 2,
        )
        theta2_colors = angle_to_argb(
            interpolate_angle(snapshots[:, 1, :], 0.0), lut, 2,
        )

        assert not np.array_equal(theta1_colors, theta2_colors)

    def test_same_angles_produce_same_colors(self):
        """Theta1 and theta2 with identical values should produce identical images."""
        lut = build_hue_lut()
        n = 4
        n_samples = 10

        snapshots = np.zeros((n, 2, n_samples), dtype=np.float32)
        snapshots[:, 0, :] = 1.0
        snapshots[:, 1, :] = 1.0

        theta1_colors = angle_to_argb(
            interpolate_angle(snapshots[:, 0, :], 0.0), lut, 2,
        )
        theta2_colors = angle_to_argb(
            interpolate_angle(snapshots[:, 1, :], 0.0), lut, 2,
        )

        np.testing.assert_array_equal(theta1_colors, theta2_colors)

    def test_angle_index_selects_correct_slice(self):
        """Indexing snapshots[:, idx, :] should select the correct angle."""
        n = 4
        n_samples = 10
        snapshots = np.zeros((n, 2, n_samples), dtype=np.float32)
        snapshots[:, 0, :] = 0.5
        snapshots[:, 1, :] = 2.5

        for idx in (0, 1):
            sliced = snapshots[:, idx, :]
            expected_val = 0.5 if idx == 0 else 2.5
            np.testing.assert_allclose(sliced, expected_val)


class TestBivariateAngleSlicing:
    """Test that both angle slices can be extracted for bivariate mode."""

    def test_both_slices_extractable(self):
        n = 4
        n_samples = 10
        snapshots = np.zeros((n, 2, n_samples), dtype=np.float32)
        snapshots[:, 0, :] = 0.5   # theta1
        snapshots[:, 1, :] = 2.5   # theta2

        theta1 = interpolate_angle(snapshots[:, 0, :], 0.0)
        theta2 = interpolate_angle(snapshots[:, 1, :], 0.0)

        np.testing.assert_allclose(theta1, 0.5)
        np.testing.assert_allclose(theta2, 2.5)

    def test_both_slices_interpolated(self):
        """Both slices should interpolate correctly at fractional time."""
        n = 2
        snapshots = np.zeros((n, 2, 3), dtype=np.float32)
        snapshots[:, 0, :] = [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
        snapshots[:, 1, :] = [[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]

        theta1 = interpolate_angle(snapshots[:, 0, :], 0.5)
        theta2 = interpolate_angle(snapshots[:, 1, :], 0.5)

        np.testing.assert_allclose(theta1, 0.5, atol=1e-5)
        np.testing.assert_allclose(theta2, 3.5, atol=1e-5)
