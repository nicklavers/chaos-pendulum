"""Tests for fractal/winding.py: extraction, colormaps, pipeline, legend."""

import math

import numpy as np
import pytest

from fractal.winding import (
    extract_winding_numbers,
    winding_direction_brightness,
    winding_modular_grid,
    winding_basin_hash,
    winding_to_argb,
    build_winding_legend,
    WINDING_COLORMAPS,
)


class TestExtractWindingNumbers:
    """Test winding number extraction from unwrapped angles."""

    def test_zero_angle_zero_winding(self):
        theta1 = np.array([0.0], dtype=np.float32)
        theta2 = np.array([0.0], dtype=np.float32)
        n1, n2 = extract_winding_numbers(theta1, theta2)
        assert n1[0] == 0
        assert n2[0] == 0

    def test_one_full_rotation(self):
        """2*pi should give winding number 1."""
        two_pi = np.float32(2.0 * math.pi)
        theta1 = np.array([two_pi], dtype=np.float32)
        theta2 = np.array([two_pi], dtype=np.float32)
        n1, n2 = extract_winding_numbers(theta1, theta2)
        assert n1[0] == 1
        assert n2[0] == 1

    def test_negative_rotation(self):
        """-2*pi should give winding number -1."""
        two_pi = np.float32(2.0 * math.pi)
        theta1 = np.array([-two_pi], dtype=np.float32)
        theta2 = np.array([-two_pi], dtype=np.float32)
        n1, n2 = extract_winding_numbers(theta1, theta2)
        assert n1[0] == -1
        assert n2[0] == -1

    def test_multiple_rotations(self):
        """3*2*pi should give winding number 3."""
        theta = np.array([6.0 * math.pi], dtype=np.float32)
        n1, _ = extract_winding_numbers(theta, theta)
        assert n1[0] == 3

    def test_partial_rotation_rounds_correctly(self):
        """Angles near half-integers of 2*pi should round correctly."""
        two_pi = 2.0 * math.pi
        # 0.3 * 2*pi -> rounds to 0
        theta_low = np.array([0.3 * two_pi], dtype=np.float32)
        n1, _ = extract_winding_numbers(theta_low, theta_low)
        assert n1[0] == 0

        # 0.6 * 2*pi -> rounds to 1
        theta_high = np.array([0.6 * two_pi], dtype=np.float32)
        n1, _ = extract_winding_numbers(theta_high, theta_high)
        assert n1[0] == 1

    def test_mixed_winding_numbers(self):
        """Array with mixed windings."""
        two_pi = 2.0 * math.pi
        theta1 = np.array([0.0, two_pi, -two_pi, 2 * two_pi], dtype=np.float32)
        theta2 = np.array([two_pi, 0.0, two_pi, -3 * two_pi], dtype=np.float32)
        n1, n2 = extract_winding_numbers(theta1, theta2)
        np.testing.assert_array_equal(n1, [0, 1, -1, 2])
        np.testing.assert_array_equal(n2, [1, 0, 1, -3])

    def test_output_dtype(self):
        theta = np.array([0.0, 1.0], dtype=np.float32)
        n1, n2 = extract_winding_numbers(theta, theta)
        assert n1.dtype == np.int32
        assert n2.dtype == np.int32


class TestWindingColormapOutputShape:
    """All colormap functions should return (N, 4) uint8 BGRA."""

    @pytest.mark.parametrize("colormap_fn", [
        winding_direction_brightness,
        winding_modular_grid,
        winding_basin_hash,
    ])
    def test_output_shape(self, colormap_fn):
        n1 = np.array([0, 1, -1, 2, -2], dtype=np.int32)
        n2 = np.array([0, -1, 1, -2, 2], dtype=np.int32)
        result = colormap_fn(n1, n2)
        assert result.shape == (5, 4)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("colormap_fn", [
        winding_direction_brightness,
        winding_modular_grid,
        winding_basin_hash,
    ])
    def test_all_opaque(self, colormap_fn):
        """Alpha channel should always be 255."""
        n1 = np.array([0, 1, -1], dtype=np.int32)
        n2 = np.array([0, -1, 1], dtype=np.int32)
        result = colormap_fn(n1, n2)
        np.testing.assert_array_equal(result[:, 3], 255)


class TestWindingColormapDiscrimination:
    """Different winding pairs should map to different colors."""

    @pytest.mark.parametrize("colormap_fn", [
        winding_direction_brightness,
        winding_modular_grid,
        winding_basin_hash,
    ])
    def test_adjacent_basins_differ(self, colormap_fn):
        """Adjacent winding numbers should produce different colors."""
        n1 = np.array([0, 1, 0, -1], dtype=np.int32)
        n2 = np.array([0, 0, 1, 0], dtype=np.int32)
        result = colormap_fn(n1, n2)

        # At least some pairs should differ
        colors = set(tuple(row) for row in result.tolist())
        assert len(colors) >= 2, "All adjacent basins have the same color"


class TestWindingToArgb:
    """Test the full pipeline function."""

    def test_output_shape_4x4(self):
        """Pipeline should produce (res, res, 4) output."""
        theta1 = np.zeros(16, dtype=np.float32)
        theta2 = np.zeros(16, dtype=np.float32)
        result = winding_to_argb(
            theta1, theta2, winding_direction_brightness, 4,
        )
        assert result.shape == (4, 4, 4)
        assert result.dtype == np.uint8

    def test_different_angles_different_pixels(self):
        """Different unwrapped angles should produce different pixel colors."""
        two_pi = np.float32(2.0 * math.pi)
        # 4 pixels: winding (0,0), (1,0), (0,1), (1,1)
        theta1 = np.array([0.0, two_pi, 0.0, two_pi], dtype=np.float32)
        theta2 = np.array([0.0, 0.0, two_pi, two_pi], dtype=np.float32)
        result = winding_to_argb(
            theta1, theta2, winding_direction_brightness, 2,
        )
        # Flatten to check uniqueness
        pixels = result.reshape(-1, 4)
        colors = set(tuple(row) for row in pixels.tolist())
        assert len(colors) >= 2


class TestBuildWindingLegend:
    """Test legend builder."""

    def test_output_shape(self):
        """Legend should be a square BGRA image."""
        legend = build_winding_legend(winding_direction_brightness, n_range=3, cell_size=6)
        expected_px = 7 * 6  # (2*3+1) * 6 = 42
        assert legend.shape == (expected_px, expected_px, 4)
        assert legend.dtype == np.uint8

    def test_all_opaque(self):
        """All alpha values should be 255."""
        legend = build_winding_legend(winding_modular_grid, n_range=2, cell_size=4)
        assert np.all(legend[:, :, 3] == 255)

    def test_small_legend(self):
        """Even with n_range=1, should produce valid output."""
        legend = build_winding_legend(winding_basin_hash, n_range=1, cell_size=2)
        expected_px = 3 * 2  # (2*1+1) * 2 = 6
        assert legend.shape == (expected_px, expected_px, 4)


class TestWindingColormapRegistry:
    """Test the WINDING_COLORMAPS registry."""

    def test_registry_has_three_entries(self):
        assert len(WINDING_COLORMAPS) == 3

    def test_all_entries_callable(self):
        for name, fn in WINDING_COLORMAPS.items():
            assert callable(fn), f"{name} is not callable"

    def test_expected_names(self):
        expected = {
            "Direction + Brightness",
            "Modular Grid (5\u00d75)",
            "Basin Hash",
        }
        assert set(WINDING_COLORMAPS.keys()) == expected

    @pytest.mark.parametrize("name", list(WINDING_COLORMAPS.keys()))
    def test_registry_functions_produce_valid_output(self, name):
        fn = WINDING_COLORMAPS[name]
        n1 = np.array([0, 1, -1], dtype=np.int32)
        n2 = np.array([0, -1, 1], dtype=np.int32)
        result = fn(n1, n2)
        assert result.shape == (3, 4)
        assert result.dtype == np.uint8
