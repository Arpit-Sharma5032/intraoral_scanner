IOSCAN session-start fix
========================

Cause
-----
Dashboard start_capture() created data/sessions/ but did not create the
individual data/sessions/<session_name>/ directory before writing roi.json.
This raised FileNotFoundError and prevented scan start.

Fix
---
The dashboard now creates the session directory itself before saving the
per-session ROI snapshot. A regression test exercises start_capture() without
opening physical camera hardware.

Apply
-----
Stop the dashboard. Extract this update archive directly into the existing
project root (the folder containing pyproject.toml), allowing
src/ioscan/dashboard/server.py and tests/test_dashboard.py to be overwritten.
Restart with:

    python scripts/run_dashboard.py

Optional verification:

    python -m pytest -q tests/test_dashboard.py

The update does not change camera, ROI, calibration, or hardware geometry
settings.
