# Hardware Geometry & Scan Settings (Dashboard)

Open **Setup → Hardware geometry & scan settings**.

The dashboard provides these editable values:

- **Camera–laser baseline (mm):** nominal separation used by the uncalibrated laser-plane model.
- **Laser tilt angle (degrees):** nominal angle used by the uncalibrated laser-plane model.
- **Calculated working distance (mm):** read-only result `baseline / tan(tilt)`. It is not an independent parameter when baseline and tilt are both specified.
- **Laser fan half-angle (degrees):** saved as a hardware setup value. The current single-plane triangulation implementation does not use it to calculate depth.
- **Scan exposure:** applied by the camera capture path during synchronized scanning.
- **Assumed scan speed (mm/s):** used by live pose tracking and saved-session reconstruction for estimated translation.
- **Minimum / maximum reconstruction depth (mm):** applied to both live point preview and final reconstruction filtering.

Click **Save settings**. The dashboard writes these values to `configs/hardware_geometry.json`. The settings are loaded automatically when the dashboard starts. They are also copied into each captured session's `meta.json`.

## Important behavior

1. Without a measured calibration JSON, the baseline and tilt define the nominal laser plane used by the live preview and captured session.
2. With a measured calibration JSON selected, its measured camera intrinsics and laser plane override the nominal baseline and tilt. The baseline/tilt fields remain stored as setup notes/defaults but do not override measured geometry.
3. The fan half-angle is not currently part of the single-plane ray/plane triangulation. It is persisted for reference and future hardware-FOV logic; changing it alone will not change the reconstructed point positions.
4. Changing assumed scan speed changes estimated translation. The IMU provides orientation, not direct linear displacement, so this remains an approximation unless an encoder or visual-odometry source is integrated.
5. All current default geometry values are nominal starting points, not measured calibration. Do not use uncalibrated output for dimensional or clinical decisions.
