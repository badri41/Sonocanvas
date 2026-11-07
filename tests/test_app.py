from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def test_app_runs_preset_and_blank():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert float(at.metric[0].value) > 0.9
    at.text_input[0].set_value("   ").run()
    assert not at.exception and at.info
