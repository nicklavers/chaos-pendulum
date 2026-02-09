"""Tests for fractal/cache.py: hit/miss/eviction/invalidation/budget."""

import numpy as np
import pytest

from simulation import DoublePendulumParams
from fractal.cache import FractalCache, CacheKey, _quantize, _params_hash
from fractal.compute import FractalViewport


def _make_key(resolution=64, center1=0.0, center2=0.0, span1=6.28, span2=6.28,
              params=None):
    """Helper to create a CacheKey."""
    if params is None:
        params = DoublePendulumParams()
    return CacheKey(
        resolution=resolution,
        center_theta1_q=_quantize(center1),
        center_theta2_q=_quantize(center2),
        span_theta1_q=_quantize(span1),
        span_theta2_q=_quantize(span2),
        params_hash=_params_hash(params),
    )


def _make_data(resolution=64, n_samples=96):
    """Create a dummy snapshot array."""
    n = resolution * resolution
    return np.random.randn(n, 2, n_samples).astype(np.float32)


class TestCacheKey:
    """Test CacheKey construction and hashing."""

    def test_from_viewport(self):
        vp = FractalViewport(
            center_theta1=1.0, center_theta2=2.0,
            span_theta1=3.0, span_theta2=4.0,
            resolution=128,
        )
        params = DoublePendulumParams()
        key = CacheKey.from_viewport(vp, params)
        assert key.resolution == 128
        assert key.center_theta1_q == _quantize(1.0)

    def test_same_viewport_same_key(self):
        params = DoublePendulumParams()
        vp = FractalViewport(1.0, 2.0, 3.0, 4.0, 128)
        k1 = CacheKey.from_viewport(vp, params)
        k2 = CacheKey.from_viewport(vp, params)
        assert k1 == k2

    def test_different_params_different_key(self):
        vp = FractalViewport(1.0, 2.0, 3.0, 4.0, 128)
        p1 = DoublePendulumParams(m1=1.0)
        p2 = DoublePendulumParams(m1=2.0)
        k1 = CacheKey.from_viewport(vp, p1)
        k2 = CacheKey.from_viewport(vp, p2)
        assert k1 != k2


class TestQuantize:
    """Test viewport coordinate quantization."""

    def test_quantize_roundtrip(self):
        """Quantized values should be stable on re-quantize."""
        val = 1.234567
        q1 = _quantize(val)
        q2 = _quantize(q1 * 1e-6)  # convert back and re-quantize
        assert q1 == q2

    def test_small_differences_collapse(self):
        """Values within one quantum should quantize the same."""
        a = _quantize(1.0000001)
        b = _quantize(1.0000002)
        # These differ by 1e-7, well within the 1e-6 quantum
        assert a == b


class TestFractalCache:
    """Test FractalCache operations."""

    def test_miss_returns_none(self):
        cache = FractalCache(max_bytes=100 * 1024 * 1024)
        key = _make_key()
        assert cache.get(key) is None

    def test_put_and_get(self):
        cache = FractalCache(max_bytes=100 * 1024 * 1024)
        key = _make_key()
        data = _make_data(64, 96)
        cache.put(key, data)
        result = cache.get(key)
        assert result is not None
        np.testing.assert_array_equal(result, data)

    def test_memory_tracking(self):
        cache = FractalCache(max_bytes=100 * 1024 * 1024)
        key = _make_key()
        data = _make_data(64, 96)
        cache.put(key, data)
        assert cache.memory_used_bytes == data.nbytes
        assert cache.entry_count == 1

    def test_eviction_when_over_budget(self):
        """Cache should evict LRU entries when over budget."""
        # Budget for ~2 entries of 64x64
        entry_size = 64 * 64 * 96 * 4  # float32
        cache = FractalCache(max_bytes=int(entry_size * 2.5))

        for i in range(4):
            key = _make_key(resolution=128, center1=float(i))  # non-protected (128)
            data = _make_data(64, 96)
            cache.put(key, data)

        # Should have evicted the oldest entries
        assert cache.entry_count < 4

    def test_protected_levels_not_evicted(self):
        """Coarse resolution entries (<=64) should survive eviction."""
        entry_size = 64 * 64 * 96 * 4
        cache = FractalCache(max_bytes=int(entry_size * 1.5))

        # Add a protected entry (resolution=64)
        protected_key = _make_key(resolution=64, center1=0.0)
        protected_data = _make_data(64, 96)
        cache.put(protected_key, protected_data)

        # Add a non-protected entry that should trigger eviction
        big_key = _make_key(resolution=128, center1=1.0)
        big_data = _make_data(64, 96)
        cache.put(big_key, big_data)

        # Protected entry should still be there
        assert cache.get(protected_key) is not None

    def test_invalidate_params(self):
        """Invalidating by params hash should remove matching entries."""
        cache = FractalCache(max_bytes=100 * 1024 * 1024)
        params1 = DoublePendulumParams(m1=1.0)
        params2 = DoublePendulumParams(m1=2.0)

        key1 = _make_key(params=params1)
        key2 = _make_key(params=params2, center1=1.0)

        cache.put(key1, _make_data())
        cache.put(key2, _make_data())
        assert cache.entry_count == 2

        cache.invalidate_params(_params_hash(params1))
        assert cache.entry_count == 1
        assert cache.get(key1) is None
        assert cache.get(key2) is not None

    def test_clear(self):
        cache = FractalCache(max_bytes=100 * 1024 * 1024)
        cache.put(_make_key(), _make_data())
        cache.put(_make_key(center1=1.0), _make_data())
        assert cache.entry_count == 2

        cache.clear()
        assert cache.entry_count == 0
        assert cache.memory_used_bytes == 0

    def test_put_validates_dtype(self):
        cache = FractalCache(max_bytes=100 * 1024 * 1024)
        key = _make_key()
        data = np.zeros((64 * 64, 2, 96), dtype=np.float64)  # wrong dtype

        with pytest.raises(ValueError, match="float32"):
            cache.put(key, data)

    def test_put_validates_shape(self):
        cache = FractalCache(max_bytes=100 * 1024 * 1024)
        key = _make_key()
        data = np.zeros((64 * 64, 96), dtype=np.float32)  # 2D, not 3D

        with pytest.raises(ValueError, match="3D"):
            cache.put(key, data)

    def test_replace_existing_entry(self):
        """Putting with same key should replace the old entry."""
        cache = FractalCache(max_bytes=100 * 1024 * 1024)
        key = _make_key()
        data1 = _make_data()
        data2 = _make_data()

        cache.put(key, data1)
        old_bytes = cache.memory_used_bytes

        cache.put(key, data2)
        assert cache.entry_count == 1
        assert cache.memory_used_bytes == old_bytes  # same size
        np.testing.assert_array_equal(cache.get(key), data2)
