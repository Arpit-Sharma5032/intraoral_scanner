# Reconstruction approach: why outputs were broken, and the demo-ready path

## Verdict on the previous approach

The scanner projects **one laser line**. Each frame therefore measures one 1-D profile of
the surface. That profile is accurate in itself: it depends only on calibration. To get a
surface, every profile must also be placed where the scanner was at that instant. The old
default (`imu_hybrid`) did this with:

- **rotation:** the absolute IMU orientation, applied rigidly to each profile;
- **translation:** a *constant assumed* 5 mm/s along the rotated camera +Y axis.

Translation is never measured, so that placement is made up. Evidence from session
`scan_20261114_181652`:

| Observation | Value |
|---|---|
| Capture length | 93.4 s, 4,608 frames |
| Assumed path (5 mm/s) | 467 mm (a full arch is about 120 mm) |
| IMU rotation away from start pose | up to 163° |
| Time with a usable laser line | about 21–68 s; little or nothing before or after |
| Mesh extent | 100 × 113 × 205 mm |

Recovering each profile from `fused_cloud.ply` in camera coordinates shows a coherent surface
only during 21–40 s. Everything else was placed along an invented 467 mm curve and then
meshed with ball pivoting. That produced the floating fragments, and MeshSegNet "found 6
teeth" in them.

**Conclusion.** With a single line, an orientation-only IMU and no translation measurement, a
free-hand full-arch reconstruction cannot work, however it is tuned. Commercial intraoral
scanners capture a full 3-D patch per frame and register patches to each other (ICP). A line
profile cannot be registered that way.

## What a line scanner can do reliably (the new default)

**Linear sweep** (`--pose-mode sweep`, the dashboard default): one straight pass over a
quadrant or sextant, with the wand orientation held fixed.

1. **Ridge extraction.** The secondary-peak guard band adapts to the measured line width, so a
   broad or haloed line is no longer rejected column by column. Reflection ratio and minimum
   brightness can be set.
2. **Depth window from the ROI.** Every ROI pixel maps to exactly one depth on the laser plane,
   so the ROI already bounds what can be seen. A manual 75–125 mm window was clipping a third
   of your line, which sat at 65–92 mm.
3. **Profile cleaning.** Spikes that disagree with the running median along the line are
   removed (reflections, enamel glints), along with fragments shorter than 8 columns.
4. **Auto-trim.** Only the longest time window with a continuous line on the surface is kept.
5. **Measured translation.** The laser-OFF frames, already captured for ambient subtraction,
   are registered to each other: phase correlation for the coarse shift, then ECC sub-pixel
   refinement against a keyframe. The image shift is scaled by the measured profile depth:
   T = −shift × Z / f. Fallbacks: a known pass length (`--scan-length-mm`), then the assumed
   speed.
6. **Height-map fusion.** Profiles are binned on a 0.15 mm grid using a robust per-cell median.
   Speckles are removed and holes up to 0.6 mm are filled.
7. **Grid meshing.** The height map is triangulated directly, giving one continuous surface
   with consistent normals. Open3D is not needed for this path.

Outputs keep the same names (`reconstructed_mesh.ply/.stl`, `fused_cloud.ply`,
`pipeline_report.json`). The report gains `translation`, `active_window`, `profile_cleaning`
and `heightmap` sections.

### Validation (synthetic session matching your hardware)

The test session in `tests/_synth_dental.py` uses your ROI, 19 mm baseline, 11° tilt and an
80 mm working distance. It adds hand speed varying between 2 and 7 mm/s, sideways drift,
0.8 s of idle at each end, and random reflections.

| Translation | Result |
|---|---|
| Optical flow (auto) | Travel 42.3 mm vs 44.2 mm true; correct tooth sizes and spacing; one connected surface |
| Constant 5 mm/s | Teeth squashed or stretched wherever the hand sped up or slowed down |

Auto-trim found the on-surface window exactly (0.8–9.8 s).

## Capture protocol for a demo

1. Use a **typodont or stone model** on a table, under diffuse room light. The OFF frames need
   visible texture for optical flow; check with `diagnose_session.py`, section [5].
2. Set **exposure** so that the line peak is about 150–220 DN without saturating (section [2]
   of the diagnostic). Your last session had a median of 77 DN.
3. Choose **one straight pass** along a quadrant (premolars to molars), about 40–60 mm, in
   10–15 s. Do not rotate the wand. Keep the line on the teeth from start to end.
4. **Mode** "Linear sweep", translation "Auto". If the diagnostic says flow is unreliable,
   measure the pass length with a ruler and use "Known scan length".
5. A slider, linear rail or motorised stage makes the demo repeatable. The pipeline does not
   need it, but it removes hand rotation.

## Dashboard

Setup → *Hardware geometry & scan settings* → **Reconstruction**: mode, translation source,
scan length, reflection ratio, minimum brightness, auto-trim and depth-from-ROI. These are
saved with the other settings.

Setup → **Reprocess a saved session**: re-runs reconstruction on frames already on disk with
the current settings. No new capture is needed.

Automatic MeshSegNet is skipped in sweep mode. The model was trained on full arches and labels
a partial patch arbitrarily. *Run / retry MeshSegNet* still lets you try it.

## Command line

```powershell
python scripts/diagnose_session.py data/sessions/<session>
python scripts/run_pipeline.py data/sessions/<session> --out data/results/<session>            # sweep, auto
python scripts/run_pipeline.py data/sessions/<session> --translation length --scan-length-mm 45
python scripts/run_pipeline.py data/sessions/<session> --pose-mode imu_hybrid                  # legacy (Open3D)
```

## Full arch in ONE continuous session (recommended)

On this hardware the scanner stays fixed and the model is moved underneath it. The camera sees
the **whole model through the window** in every laser-OFF frame. The pipeline now tracks the
model's full in-plane motion (rotation + translation) from those frames (`PlanarTracker` in
`sweep.py`). Profiles from one session therefore stay in one consistent frame, however the model
is slid and turned. No stitching is needed.

How it works:
- **Finding the model.** A motion mask selects the pixels that see the moving model. It excludes
  everything fixed to the camera: the housing, the window frame, shadow edges and streaks on the
  glass. Those stay sharp in the temporal median image.
- **Tracking.** ECC (Euclidean) registration against keyframes, at quarter resolution.
  Implausible jumps between frames are rejected.
- **Placement.** The image rotation and translation are converted to millimetres at the measured
  laser depth. Each profile is placed in the model frame as X0 = R(θ)ᵀ (X − D).
- **Trimming.** Every profile that saw the surface is used, with no window trimming, because the
  line may leave the teeth briefly while the model turns.
- **Table strip.** A table or floor strip seen past the model edge (a separate depth population
  behind an empty gap) is removed before fusion.
- **Validation.** On session scan_20261114_234110, 99.7% of frames were tracked (median ECC 0.988,
  about 20–24° of turn). Ambient frames warped into one image form a coherent model, and the
  reconstructed patch projects onto the tooth row it came from.

### How to capture

1. Keep the room light steady. The OFF frames must show the model, at about 40–80 DN.
2. Set the frame limit to 4000–5000 frames (80–100 s).
3. Slide the model so the line moves from the last molar on one side, around the front teeth,
   to the last molar on the other side. Turn the model gradually as you go, so the line stays
   across the tooth row.
4. Move slowly and smoothly: turn at most about 20°/s, slide at about 1–2 mm/s.
5. Keep hands and fingers out of the window view. Hold the model by its base, outside the
   window, or slide it on a sheet.
6. The model may leave the line briefly; that's fine. It must not leave the window.
7. Processing then gives one full-arch surface. Run MeshSegNet from the Results tab.

## Full arch: several straight passes, stitched (fallback)

Stitching needs **two or more shared teeth between consecutive passes**. The five passes of
2026-11-14/15 shared about one tooth each, on uniform stone, and the matcher could not tell the
placements apart. It now reports such joints as *ambiguous* and skips MeshSegNet on them. When a
pass has an ambient reference image, image matching of the whole visible model is tried first.
It succeeds when the passes look alike, such as similar lighting and a moderate turn.

One straight pass gives an accurate patch. A full arch is scanned as **4–6 overlapping straight
passes** and stitched (`src/ioscan/pipeline/stitch.py`).

### How to capture

1. Keep everything that made the single pass good: room light (background about 40–80 DN),
   laser focus, the ROI on the tooth side, and the line reading *In range*.
2. Plan the segments around the U, in order. For example:
   1. left molars → premolars
   2. left premolars → canine
   3. canine → canine (front teeth)
   4. right canine → premolars
   5. right premolars → molars
3. **Overlap each pass with the previous one by one or two teeth.** Start each pass on the last
   tooth or two of the previous pass. Stitching matches tooth shapes in that overlap, so a pass
   with no overlap cannot be placed.
4. During a pass, keep the wand orientation fixed and translate straight along the segment. Turn
   the wand (or the model) only *between* passes, so the line stays roughly across the arch.
5. Use 1500–2000 frames per pass, at a steady 1–2 mm/s. Keep the wand pointing straight down at
   the teeth in every pass. Side views at very different angles do not fit one occlusal surface.
6. Each pass is reconstructed automatically in sweep mode after capture.

### How to stitch

**Dashboard:** Setup → **Full arch: stitch passes**. Tick the passes in scan order (the badges
show the order), name the arch, then click *Stitch selected passes*. The result opens in the
viewer. MeshSegNet runs afterwards if "Segment teeth after stitching" is ticked; it uses the jaw
and model folder from the Segmentation section.

**Command line:**
```powershell
python scripts/stitch_sessions.py scan_A scan_B scan_C scan_D scan_E --segment --jaw lower
```

### How it works

- **Coarse alignment.** Each pass is matched only against the previous pass. The search covers
  all yaw angles and uses correlation of high-pass-filtered height maps. Filtering removes the
  smooth gum and base, which fit anywhere, so tooth relief drives the match. Matching only
  against the previous pass avoids snapping onto look-alike molars on the other side.
- **Candidate check.** The top candidates are refined with point-to-plane ICP. The one whose tooth
  relief agrees best wins.
- **Global refinement.** Every pass is re-aligned to all the others, three times, to spread the
  error around the arch instead of letting it build up at the last pass.
- **IMU.** The IMU's turn angle between passes is used only as a tie-breaker and a cross-check.
  The report warns if geometry and IMU disagree by more than 20°. It never sets positions, so
  gyro drift and the unknown IMU mounting add no error.
- **Fusion.** The passes are fused into one height map along the mean viewing direction and
  trimmed to the teeth plus 5 mm of gum (the dashboard option or `--keep-base` keeps everything).
  The result is written in the project's canonical dental frame: arch in XY, teeth toward +Z,
  incisors toward +Y, in mm.
- **Outputs.** The arch's results folder holds `reconstructed_mesh.ply/.stl`, `fused_cloud.ply`
  and `pipeline_report.json` (per joint: relief match, overlap, RMS, turn angle and the IMU's
  angle).

### Validation (synthetic 14-tooth arch, 6 passes)

Each synthetic pass had ±2% movement-scale error, ±3° wand tilt and 0.03 mm noise. Over 5
random trials:
- Every pass was placed within 0.9° of its true orientation.
- The worst median position error was 0.76 mm, at the far end of the U.
- The fused surface was within 0.10 mm of the true arch (median); the 99th percentile was 0.58 mm.

### MeshSegNet caveat

The stitched mesh covers the occlusal side only (tooth walls appear as steep surfaces) and has
no undercuts. MeshSegNet was trained on full intraoral scans and is sensitive to orientation.
Labels on this mesh are a demonstration, not a clinical result. If teeth are labelled with
mirrored or shifted classes, the orientation convention (incisors toward +Y) is the first thing
to check.

## Still not solved in software

- **Full arch.** A full arch needs either several straight passes stitched together (future
  work: register overlapping height maps), or a projector or camera setup that captures a 3-D
  patch per frame.
- **Rotation during a pass.** It is ignored in sweep mode, and the report warns when the IMU
  shows more than 15°. Optical flow cannot tell small rotations from translation.
- **Calibration.** It is still nominal (19 mm, 11°). Absolute dimensions are only as good as
  that plane, so run `scripts/run_calibrate.py` before quoting millimetres.
