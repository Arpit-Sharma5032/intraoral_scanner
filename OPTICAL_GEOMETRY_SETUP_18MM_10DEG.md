# Current optical geometry setup (provisional)

This project copy has been updated to match the user's current stated hardware values:

- Camera: Arducam B0322 / OV2311, 1600 x 1200, 3 µm pixel pitch, nominal EFL 2.8 mm.
- Nominal camera focal estimate: fx = fy = 2.8 / 0.003 = 933.3 px.
- Laser baseline parameter: 18 mm.
- Laser tilt parameter: 10 degrees.
- Central-axis laser-plane intersection: Z = 18 / tan(10°) = 102.08 mm.
- Provisional camera-Z reconstruction filter: 75–125 mm.

## Important optical caveat

The Arducam B0322 datasheet lists both EFL 2.8 mm and 75° horizontal FOV. With 1600 pixels and 3 µm pitch, a thin-lens pinhole estimate from EFL gives fx≈933 px (about 81.2° horizontal FOV), while 75° FOV implies fx≈1043 px. Therefore these are nominal specifications, not a consistent measured intrinsic matrix. Run checkerboard calibration using the final lens, focus, and mirror assembly and select that calibration JSON in the dashboard. The measured JSON is authoritative and overrides this nominal model.

## Mirror-folded working distance

The geometry model's camera Z is referenced to the camera's effective optical centre / virtual camera after accounting for the mirror. For a single flat folding mirror, the approximate optical path to a tooth is:

`camera optical centre -> mirror` + `mirror -> tooth`

If the 64 mm measurement is from the camera effective optical centre to the mirror, then the 102.08 mm central intersection corresponds to about 38.1 mm from mirror to tooth. If 64 mm was measured from the camera housing or sensor board, correct for the lens principal plane/entrance-pupil offset. If the mirror is not a simple flat fold mirror, or camera and laser do not share the same reflected geometry, do not use this arithmetic as calibration; fit the camera intrinsics and laser plane experimentally through the actual optical path.

## Depth window

75–125 mm is a commissioning window only. With a 64 mm camera-to-mirror optical segment, it accepts approximately 11–61 mm beyond the mirror. Narrow it after measuring the real tooth position and checking detected laser-line pixels. Do not use it for clinical/dimensional measurements.

## Laser line not detected

The ridge extractor thresholds image contrast and saturation before 3D depth filtering. A wrong Z range cannot, by itself, make the 2D ridge detector report no line; it can reject all triangulated points after a line is found. A long working distance can reduce brightness or focus/sharpness and therefore indirectly cause no line detection. Check ON-OFF difference image, focus at the target distance, laser exposure/power, reflections, ROI and ON/OFF pairing.
