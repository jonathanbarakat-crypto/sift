"""Shared pytest fixtures."""

import pytest

from tests.fixtures import make_f_pair


@pytest.fixture(scope="module")
def f_pair():
    return make_f_pair()
