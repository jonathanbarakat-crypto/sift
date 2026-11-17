"""Shared fixtures for the sim harness tests.

SIMULATED — infrastructure validation, not user-behavior evidence.

Every test here runs with SIFT_ENV=test against an ephemeral tmp_path DB.
The flags are reset around each test (beta default: cards_v4_1 ON).
"""

import os

import pytest

from sift import flags

os.environ["SIFT_ENV"] = "test"
os.environ.setdefault("SIFT_CARD_HMAC_SECRET", "sim-test-hmac-secret")


@pytest.fixture(autouse=True)
def _flags_reset():
    flags.reset()
    yield
    flags.reset()


@pytest.fixture()
def sim_env(monkeypatch):
    """Force the test env for the sim package's assert_test_env guard."""
    monkeypatch.setenv("SIFT_ENV", "test")
    return "test"
