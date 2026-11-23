from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_agent_mode():
    """Default all tests to mock agent mode.

    Patches the env var so even if another test reloads app.config
    (creating a fresh setting object), the new object still picks
    up AGENT_MODE=mock.
    """
    with patch.dict(os.environ, {"AGENT_MODE": "mock"}):
        import app.config
        original = app.config.setting.agent_mode
        app.config.setting.agent_mode = "mock"
        yield
        app.config.setting.agent_mode = original
