"""Execute the public Streamlit page and its primary interaction flow.

This is intentionally a plain script instead of a pytest test: the deployment
CI job installs exactly ``requirements.txt``, matching Community Cloud, without
the development test stack.

The two serving artifacts are hosted rather than committed, so a bare checkout
does not have them. Rather than downloading them in CI, this skips with a clear
message when they are absent -- reporting a pass it did not run would be worse
than reporting nothing. It is a full check wherever the artifacts exist, which
is any development machine and any deployed container.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ARTIFACTS = (
    PROJECT_ROOT / "data" / "processed" / "als_production_item_factors.npz",
    PROJECT_ROOT / "data" / "processed" / "anime_catalog_serving.json",
)


def _assert_clean(app: AppTest, phase: str) -> None:
    if app.exception:
        details = "; ".join(str(exception.value) for exception in app.exception)
        raise AssertionError(f"Streamlit raised during {phase}: {details}")
    if app.error:
        details = "; ".join(str(error.value) for error in app.error)
        raise AssertionError(f"Streamlit rendered an error during {phase}: {details}")


def main() -> int:
    missing = [path.name for path in REQUIRED_ARTIFACTS if not path.exists()]
    if missing:
        print(f"streamlit smoke skipped: {', '.join(missing)} not present in this checkout", file=sys.stderr)
        return 0

    app = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py").run(timeout=30)
    _assert_clean(app, "initial render")

    example = next(button for button in app.button if button.label == "Sci-Fi / Psychological")
    example.click().run(timeout=30)
    _assert_clean(app, "example-profile selection")
    if len(app.session_state["liked"]) != 3:
        raise AssertionError("The example profile did not populate three liked titles")

    recommend = next(button for button in app.button if button.label == "Recommend")
    recommend.click().run(timeout=30)
    _assert_clean(app, "recommendation")

    result = app.session_state["results"]
    if result is None or len(result.items) != 12:
        raise AssertionError("The primary flow did not render 12 recommendations")
    captions = [str(caption.value) for caption in app.caption]
    if not any("18,064 catalog titles" in value and "300-candidate pool" in value for value in captions):
        raise AssertionError("The result trace does not describe the verified full-catalog run")

    print("streamlit smoke passed: initial render, profile selection, 12 recommendations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
