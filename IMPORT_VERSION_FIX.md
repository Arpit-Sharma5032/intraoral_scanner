# Stale-installation guard

If a stale `ioscan` copy in `site-packages` shadows this checkout, the scripts could fail with errors
such as `TypeError: PoseTracker() got an unexpected keyword argument 'initial_timestamp_s'`.

Every CLI script in `scripts/` now puts this checkout's `src/` first on `sys.path` before importing
`ioscan` (`run_pipeline`, `run_segment`, `run_capture`, `view_results`, `run_eval`, `run_sim`,
`check_session`, `inspect_mesh`, `benchmark_triangulation`; `run_calibrate` inherits it by importing
`run_pipeline`).

Verify which copy is used:

```powershell
python -c "import ioscan.pipeline.pose as p, inspect; print(p.__file__); print(inspect.signature(p.PoseTracker))"
```

For interactive use, refresh the editable install: `python -m pip install -e . --no-deps --force-reinstall`.
