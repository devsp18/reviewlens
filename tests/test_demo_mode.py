"""Demo Mode must never trigger a Gemini call - these tests assert the UI buttons
that would spend API credits stay disabled, and that classify_reviews is never
invoked (patched to raise if called), regardless of what's clicked."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from reviewlens.config import settings

APP_DIR = Path(__file__).resolve().parent.parent / "app"


def test_dashboard_classify_button_disabled_in_demo_mode(monkeypatch, mocker):
    monkeypatch.setattr(settings, "demo_mode", True)
    mock = mocker.patch("reviewlens.classify.classify_reviews", side_effect=AssertionError("must not be called"))

    at = AppTest.from_file(str(APP_DIR / "pages" / "1_📊_Dashboard.py"))
    at.run(timeout=30)

    classify_buttons = [b for b in at.button if "Classify" in b.label]
    assert classify_buttons and all(b.disabled for b in classify_buttons)
    mock.assert_not_called()


def test_backlog_rebuild_button_disabled_in_demo_mode(monkeypatch, mocker):
    monkeypatch.setattr(settings, "demo_mode", True)
    mock = mocker.patch("reviewlens.cluster.build_pain_points", side_effect=AssertionError("must not be called"))

    at = AppTest.from_file(str(APP_DIR / "pages" / "2_🎯_Backlog.py"))
    at.run(timeout=30)

    pain_point_buttons = [b for b in at.button if "pain points" in b.label]
    assert pain_point_buttons and all(b.disabled for b in pain_point_buttons)
    mock.assert_not_called()


def test_effective_db_path_switches_to_demo_db(monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", True)
    assert settings.effective_db_path == settings.demo_db_path
    monkeypatch.setattr(settings, "demo_mode", False)
    assert settings.effective_db_path == settings.db_path
