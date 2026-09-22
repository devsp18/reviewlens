"""Shared pytest fixtures - all tests use an isolated tmp SQLite DB, never the real one."""

from pathlib import Path

import pytest

from reviewlens import db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    db.init_db(path)
    return path
