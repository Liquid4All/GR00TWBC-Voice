"""Pytest configuration for the voice_control test suite.

Ensures the repository root is importable so ``import voice_control`` works
regardless of how pytest is invoked (``pytest tests/`` vs ``python -m pytest``).
Without this, pytest's default ``prepend`` import mode only puts the ``tests/``
directory on ``sys.path``, not the repo root where the ``voice_control`` package
lives.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
