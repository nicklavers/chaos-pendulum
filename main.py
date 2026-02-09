"""Entry point for the Chaos Pendulum application.

Supports two modes:
- Pendulum: animate double pendulum trajectories
- Fractal Explorer: 2D state-space fractal visualization
"""

import logging
import sys

from PyQt6.QtWidgets import QApplication

from app_window import AppWindow


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()

    # Numba JIT warmup in background (if available)
    _warmup_numba()

    sys.exit(app.exec())


def _warmup_numba():
    """Trigger Numba JIT compilation in background if available."""
    try:
        from fractal._numba_backend import NumbaBackend
    except ImportError:
        return

    from PyQt6.QtCore import QThread

    class WarmupThread(QThread):
        def run(self):
            NumbaBackend.warmup()
            logging.getLogger(__name__).info("Numba JIT warmup complete")

    # Store reference to prevent GC
    _warmup_numba._thread = WarmupThread()
    _warmup_numba._thread.start()


if __name__ == "__main__":
    main()
