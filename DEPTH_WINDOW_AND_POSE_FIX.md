# Broken point cloud despite >100k points: causes and fixes

> **Superseded — see `RECONSTRUCTION_APPROACH.md`.** Point 1 below was diagnosed from an older `roi.json`;
> the real session's ROI maps to 53-129 mm and overlaps the window. The dominant cause was the pose model
> (assumed translation + IMU rotation), now replaced by the default linear-sweep reconstruction.

## What the failing session (scan_20261114_181652) shows
1. **Depth window vs ROI mismatch (main cause).** With the stored nominal laser plane the saved ROI
   (image rows ~651-1174) maps to Z = 24-76 mm, but the depth gate was 75-125 mm. Only ~0.05 % of ROI
   pixels (about 50 of 692 columns) can ever pass, which matches the reported 65.8 points/profile
   (a 1600-px line should give hundreds). The surviving points are a sliver at the gate edge, so the
   cloud is not the tooth surface. 107k points = 1682 profiles x ~65 points, not a dense scan.
2. **Open-loop translation.** 4608 frames at 5 mm/s = 467 mm of assumed path (mesh extent 100x113x205 mm).
   A dental pass is ~50-100 mm, so the real average speed was several times lower and profiles were
   stretched into a ribbon. The IMU gives rotation only, and at ~50-100 mm range 1 deg = ~1-1.7 mm.
3. **The quality gate accepted it.** `extent >= 35 mm` passed a 205 mm ribbon, so MeshSegNet "found
   6 teeth" on garbage.

## Changes
- `triangulate.py`: `roi_depth_coverage`, `resolve_depth_window` (ROI-derived depth window).
- `run_pipeline.py`: auto-replaces a non-overlapping window (warning + report field `depth_window`;
  `--strict-depth-window` to fail instead), `--scan-length-mm L` (speed = L / duration),
  `--imu-to-camera-deg RX RY RZ`, new mesh plausibility gate.
- `quality.py` (new): extent upper bound (110 mm), connectivity, path plausibility.
- `segment.py`: refuses meshes larger than an arch.
- `pose.py`: optional IMU->camera extrinsic.
- `dashboard/server.py`: live preview uses the same ROI-derived window.
- `scripts/diagnose_session.py` (new): shows where the real laser line sits, which gate removes points,
  IMU rotation, implied path.

## Not solved in software
Real handheld translation. Use `--scan-length-mm` with a known travel, or a motorised stage/encoder.
Calibration is still nominal; depths are only as good as the nominal plane (measure it).
