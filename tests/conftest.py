"""Pytest configuration and shared fixtures.

Fixtures defined here are available to all tests without explicit import.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


# -- Paths ----------------------------------------------------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Path to the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def golden_dir(repo_root: Path) -> Path:
    """Directory holding golden-master data for regression tests."""
    p = repo_root / "tests" / "golden"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    """Directory holding small test fixtures (tiny synthetic data)."""
    p = repo_root / "tests" / "fixtures"
    p.mkdir(parents=True, exist_ok=True)
    return p


# -- Synthetic data -------------------------------------------------------


@pytest.fixture
def synthetic_extension() -> np.ndarray:
    """A clean extension axis, 0 → 300 nm in 5000 points (matches old default)."""
    return np.linspace(0.0, 300.0, 5000)


@pytest.fixture
def synthetic_wlc_force(synthetic_extension: np.ndarray) -> np.ndarray:
    """A synthetic WLC force curve with p=0.4 nm, L=200 nm, no noise."""
    p = 0.4
    lc = 200.0
    x = synthetic_extension
    # Marko-Siggia; same formula used throughout the original Igor code.
    return (4.1 / p) * (0.25 * (1.0 - x / lc) ** -2 - 0.25 + x / lc)


@pytest.fixture
def synthetic_force_curve(
    synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
) -> ForceCurve:  # type: ignore[name-defined]  # noqa: F821
    """A noise-free synthetic ForceCurve used by many unit tests."""
    from afmkit.core.curve import ForceCurve

    return ForceCurve(
        extension=synthetic_extension,
        force=synthetic_wlc_force,
        metadata={"k_cantilever": 0.1, "temperature": 298.0, "source": "synthetic"},
    )
