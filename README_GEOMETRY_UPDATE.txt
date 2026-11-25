IOSCAN Hardware Geometry Dashboard Update

1. Stop the IOSCAN dashboard process.
2. Back up your existing project folder.
3. Extract this archive into the project root (the folder containing pyproject.toml), allowing files to be overwritten.
4. Activate the project's ios_scan environment.
5. Restart with: python scripts/run_dashboard.py
6. Open Setup -> Hardware geometry & scan settings.
7. Adjust values, then click Save settings.

The dashboard stores settings in configs/hardware_geometry.json.
A measured Calibration JSON overrides the nominal baseline and laser tilt.
The displayed working distance is calculated from baseline / tan(tilt).
The fan half-angle is currently saved for reference; this single-plane triangulation implementation does not use it yet.
All default optical geometry values are nominal, not measured calibration.
