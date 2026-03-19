"""Shared fixtures for agent-wrangler tests."""
import json
import pytest


@pytest.fixture
def sample_projects_config(tmp_path):
    """Create a temporary projects.json with test data."""
    config = {
        "projects": [
            {"id": "webapp", "name": "webapp", "path": "/home/user/webapp", "default_branch": "main"},
            {"id": "api", "name": "api", "path": "/home/user/api", "default_branch": "main", "group": "business"},
            {"id": "docs", "name": "docs", "path": "/home/user/docs", "default_branch": "main", "barn": True},
        ]
    }
    config_file = tmp_path / "projects.json"
    config_file.write_text(json.dumps(config))
    return config_file, config


@pytest.fixture
def sample_store(tmp_path):
    """Create a temporary team_grid.json with test data."""
    store = {
        "default_session": "test-grid",
        "default_layout": "tiled",
        "default_projects": ["webapp", "api"],
        "persistence": {"enabled": False, "autosave_minutes": 15, "last_snapshot": ""},
        "profiles": {
            "current": "default",
            "items": {"default": {"managed_sessions": [], "max_panes": 10}},
        },
    }
    store_file = tmp_path / "team_grid.json"
    store_file.write_text(json.dumps(store))
    return store_file, store
