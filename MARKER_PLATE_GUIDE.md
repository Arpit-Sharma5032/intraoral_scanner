# Marker plate, calibration and scanning

The scanner measures one laser profile per camera frame. To place the profiles it has to know
how the model moved. A printed sheet of markers under the model tells it exactly that, in every
frame: position, turn, tilt and height. This replaces guessing the motion from the look of the
white model, which was the main reason scans failed or the arch came out bent.

The same guide is in the dashboard: press **Guide** in the top bar.

## 1. Prepare the two sheets (once)

1. Open `docs/marker_sheet_A4.pdf` (dashboard: Setup → Calibration → Open PDF).
2. Print **two copies at 100 %** ("Actual size", not "Fit to page"), on matt paper.
3. Measure the **100 mm bar** on the print with a ruler. Note the length (for example 99.6 mm).
4. Cut along the dashed line. Glue each copy **flat on a stiff card** (thick cardboard, acrylic,
   a book cover). No tape, lamination or glossy film over the markers.
5. Copy A stays bare. It is the **calibration sheet**.
6. Copy B is the **model plate**. Fix the model on its middle with tack or double-sided tape so
   that it cannot shift. Markers under the model are simply not used.

## 2. Calibrate (once; again only if the camera, lens or laser is moved)

1. Start the dashboard and press **Check camera**.
2. Put the bare calibration sheet under the scanner, where the model will be.
3. Open **Setup → Calibration**. Enter the measured bar length.
4. Press **Record calibration (30 s)**. While it records, move the sheet slowly:
   - **slide** it around so different parts of it pass under the camera,
   - **lift** the whole sheet 1–2 cm and lower it again,
   - **tilt** it by lifting each edge in turn (15–25°).
   Keep the laser line on the sheet the whole time.
5. Wait for **Calibration saved**. A sheet residual below 0.15 mm is good.
6. If the message says the sheet was not tilted enough, or the line was always at the same
   height, record again with larger tilts and lifts. The previous calibration is kept until a new
   one succeeds.

The result is stored in `data/calibration/calibration.json` and used for every new scan and for
Reprocess. A scan made on the marker plate **before** calibrating does not need to be repeated:
select it in the Library and press Reprocess.

## 3. Scan

1. Put the model plate under the scanner. In the **Scan** tab, the **Ready check** shows how
   many markers the camera sees. Four or more is good; two is the minimum.
2. Press **Start scan**.
3. Slide and turn the plate so the laser line crosses every tooth, from the last molar on one
   side, around the front teeth, to the last molar on the other side.
4. Press **Stop**. The result is marked **Good**, **Usable** or **Rescan**, with the reason.

What you may do: move at any speed, stop, go back over a part, scan in several passes.

What to keep in mind:
- **Keep the plate flat on the floor of the box and slide it.** This is the most accurate way,
  and it works even when the model hides most of the sheet. Tilting or lifting during a scan is
  measured too, but only accurately while many markers (six or more) are in view.
- **Keep at least two markers visible** and your fingers off the print. Frames without markers
  are skipped, not guessed, so they leave a gap rather than a distortion.
- The model must not move on the plate.

## 4. Tooth sides, and combining scans

A laser seen from above cannot reach the steep side walls of the teeth.

1. Rest the plate on a wedge of about 30° so the cheek side of the teeth faces the laser, and
   scan again, sliding the plate on the wedge.
2. Turn the plate round on the wedge and scan the tongue side the same way.
3. In the **Library**, tick the scans made on the same plate and press **Build arch**. They are
   combined directly in the plate's coordinates. Nothing is stitched, so nothing can be stitched
   wrongly, as long as the model stayed fixed on the plate.

Build arch also works on a single marker-plate scan: it trims it to the teeth, turns it into the
standard dental orientation and can run MeshSegNet.

## 5. Lighting

- Shade the scanner from direct room light. Keep one small, steady lamp near it.
- **Background light** in the Ready check should be below about 90. Above 120 the laser line is
  lost on white teeth.
- The markers are black on white, so they are read in quite dim light. The plain model no longer
  needs to be lit for tracking.
- Glossy, translucent plastic teeth scatter the laser. A matte scanning spray (or a plaster
  model) gives a much stronger and sharper line.

## If something goes wrong

| Message | What to do |
|---|---|
| Marker plate: 0 markers | The sheet is out of view, in shadow, or glossy. Slide it into view; add a little light. |
| The scanner is not calibrated | Run the calibration in section 2, then Reprocess the scan. |
| Markers were lost in N % of the frames | Fingers or the model covered the print, or the plate left the camera window. |
| Only N markers were in view … heights are uncertain | Slide the plate flat on the floor instead of holding it, or leave more of the sheet visible. |
| The surface is flat | The laser ran over the base, not the teeth. Reposition so the line crosses the tooth row. |
| The laser line is too weak | Too much background light, or the line is out of focus. |
| Scan X disagrees with the earlier scans | The model moved on the plate between scans. Fix it down and rescan. |

## Technical notes

- Sheet layout: 16 × 14 markers, 6 mm, 7.5 mm pitch, dictionary DICT_4X4_250
  (`configs/marker_board.json`; regenerate the PDF with `python scripts/make_marker_sheet.py`).
- Motion source can be forced in Setup → Advanced. "Auto" uses the marker plate when its markers
  are visible and falls back to image tracking of the model otherwise.
- Command line: `python scripts/run_pipeline.py <session dir> --out <dir> --translation markers
  --calibration data/calibration/calibration.json`.
