from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def test_app_runs_preset_and_blank():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert float(at.metric[0].value) > 0.9
    at.text_input[0].set_value("   ").run()
    assert not at.exception and at.info


def test_presentation_mode_shows_auc_and_presets():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert at.metric[0].label.startswith("AUC") and float(at.metric[0].value) > 0.99
    assert not at.file_uploader and not at.radio
    assert [b.label for b in at.button[:3]] == ["DA321", "HELLO", "IITG"]


def test_presentation_falls_back_to_prerendered_wav(monkeypatch):
    monkeypatch.setenv("SONOCANVAS_FALLBACK_S", "0")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.toggle[0].set_value(True).run()
    # NOTE: HELLO wasn't encoded yet in this process, so the timeout fires and we get the fallback.
    at.button[1].click().run()
    assert not at.exception
    assert "pre-rendered demo/HELLO.wav" in at.caption[-1].value
    assert float(at.metric[0].value) > 0.99


def test_keep_font_size_lengthens_clip():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle[1].set_value(True).run()
    at.text_input[0].set_value("DRAWING WITH SOUND").run()
    assert not at.exception
    clip = float(at.caption[-1].value.split("Clip length ")[1].split(" s")[0])
    assert clip > 10 and float(at.metric[0].value) > 0.99
