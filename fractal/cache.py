"""Fractal cache: LRU cache with adaptive memory budget.

Stores computed angle snapshot arrays keyed by quantized viewport
coordinates and physics parameter hash. Supports protected coarse
levels that are never evicted.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from simulation import DoublePendulumParams

logger = logging.getLogger(__name__)

# Quantization quantum for viewport coordinates (radians)
VIEWPORT_QUANTUM = 1e-6


def _quantize(value: float) -> int:
    """Quantize a float to integer units of VIEWPORT_QUANTUM."""
    return round(value / VIEWPORT_QUANTUM)


def _params_hash(params: DoublePendulumParams) -> int:
    """Compute a stable hash for physics parameters."""
    return hash((params.m1, params.m2, params.l1, params.l2, params.g, params.friction))


@dataclass(frozen=True)
class CacheKey:
    """Immutable key for fractal cache lookups.

    All viewport coordinates are quantized to VIEWPORT_QUANTUM to avoid
    float equality issues.
    """

    resolution: int
    center_theta1_q: int
    center_theta2_q: int
    span_theta1_q: int
    span_theta2_q: int
    params_hash: int

    @classmethod
    def from_viewport(
        cls,
        viewport,
        params: DoublePendulumParams,
    ) -> CacheKey:
        """Create a CacheKey from a FractalViewport and params."""
        return cls(
            resolution=viewport.resolution,
            center_theta1_q=_quantize(viewport.center_theta1),
            center_theta2_q=_quantize(viewport.center_theta2),
            span_theta1_q=_quantize(viewport.span_theta1),
            span_theta2_q=_quantize(viewport.span_theta2),
            params_hash=_params_hash(params),
        )


class FractalCache:
    """LRU cache for fractal angle snapshot arrays.

    Features:
    - Adaptive memory budget (configurable per backend)
    - Protected coarse levels (resolution <= 64) never evicted
    - Shape/dtype validation on put()
    - Param-based invalidation
    """

    # Protected resolution threshold
    PROTECTED_RESOLUTION = 64

    def __init__(self, max_bytes: int = 512 * 1024 * 1024):
        self._max_bytes = max_bytes
        self._cache: OrderedDict[CacheKey, np.ndarray] = OrderedDict()
        self._current_bytes = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int) -> None:
        self._max_bytes = value
        self._evict_if_needed()

    @property
    def memory_used_bytes(self) -> int:
        return self._current_bytes

    @property
    def memory_used_mb(self) -> float:
        return self._current_bytes / (1024 * 1024)

    @property
    def entry_count(self) -> int:
        return len(self._cache)

    def get(self, key: CacheKey) -> np.ndarray | None:
        """Look up cached snapshots. Returns None on miss."""
        if key not in self._cache:
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: CacheKey, data: np.ndarray) -> None:
        """Store snapshot data in the cache.

        Validates shape and dtype. Evicts LRU entries if over budget.
        """
        # Validate
        if data.ndim != 3:
            raise ValueError(
                f"Cache data must be 3D (N, 2, n_samples), got shape {data.shape}"
            )
        if data.dtype != np.float32:
            raise ValueError(
                f"Cache data must be float32, got {data.dtype}"
            )

        entry_bytes = data.nbytes

        # Remove existing entry if present
        if key in self._cache:
            old_data = self._cache.pop(key)
            self._current_bytes = self._current_bytes - old_data.nbytes

        # Insert new entry
        self._cache[key] = data
        self._current_bytes = self._current_bytes + entry_bytes

        # Evict if over budget
        self._evict_if_needed()

    def invalidate_params(self, params_hash_value: int) -> None:
        """Remove all entries matching a specific params hash."""
        keys_to_remove = [
            k for k in self._cache if k.params_hash == params_hash_value
        ]
        for k in keys_to_remove:
            data = self._cache.pop(k)
            self._current_bytes = self._current_bytes - data.nbytes

    def clear(self) -> None:
        """Remove all cache entries."""
        self._cache.clear()
        self._current_bytes = 0

    def _evict_if_needed(self) -> None:
        """Evict LRU (non-protected) entries until under budget."""
        while self._current_bytes > self._max_bytes and self._cache:
            # Find the oldest non-protected entry
            evicted = False
            for key in list(self._cache.keys()):
                if key.resolution <= self.PROTECTED_RESOLUTION:
                    continue  # Never evict coarse levels
                data = self._cache.pop(key)
                self._current_bytes = self._current_bytes - data.nbytes
                logger.debug(
                    "Evicted cache entry res=%d (%.1f MB freed)",
                    key.resolution,
                    data.nbytes / (1024 * 1024),
                )
                evicted = True
                break

            if not evicted:
                # Only protected entries remain; can't evict further
                break
