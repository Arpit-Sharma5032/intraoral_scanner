# Revision 3 (code review fixes)

- **Dependencies**: segmentation no longer needs `vedo`/`plyfile` (Open3D + NumPy only); PyTorch is the
  `[seg]` extra and is imported lazily, so reconstruction works without it.
- **Segmentation**: no hard-coded user paths; explicit `--ckpt` must exist; no arbitrary
  `candidates[0]` fallback; jaw read from `meta.json`; the network now uses the reference MeshSegNet
  layer names (no input STN) and loads with a clear key-mismatch error; `weights_only=True` first;
  no silent "whole mesh as teeth" fallback; per-face colors without vertex bleeding; stale outputs
  removed; partial-scan and orientation warnings.
- **Pipeline**: reads `imu.csv` (slerp) and `t_ns`; passes `initial_timestamp_s`; no ground-truth
  rotation/velocity leakage and no default quaternion; `--speed-mm-s`, `--world-velocity`,
  `--init-from-gt`, `--z-min/--z-max`; fails instead of using nominal calibration; report restored with
  calibration, warnings and trajectory length; 8-bit saturation limit for hardware frames.
- **Capture logger**: fixed the missing `subtract_frames` import and the ridge unpacking bug; nominal
  laser plane now matches the simulator convention; writes the shared session format, `imu.csv` and
  `meta.json`; MCU clock only; waits for the first IMU packet.
- **Evaluation**: uses `gt/reference_mesh.ply` by default (previously a circular reference); normals
  oriented toward the true camera positions. Old `eval_report.json` values in `data/results` were
  produced with the circular reference and should be regenerated.
- Scripts have the `src/` path bootstrap; `configs/pipeline.yaml` marked reference-only.

At the time of the original reconstruction review, the end-to-end Open3D and MeshSegNet path had
not yet been exercised in that environment. The updated MeshSegNet implementation and the current
validation status are recorded in the integration section below.

---

# Reconstruction fixes and validation notes

## Fixed in this revision

1. **Curved-path pose integration.** The old `imu_hybrid` path inferred one fixed world velocity from
   the net start/end displacement. That ignores the arch curvature. The revised tracker integrates
   the configured camera-frame scan axis transformed by the timestamped IMU rotation, using
   trapezoidal direction integration. On the included simulated session, its 100 ON-frame camera
   positions match ground truth with median error about 0.0008 mm and maximum error about 0.0013 mm.
2. **Unconstrained profile ICP disabled.** `submap_icp` was not a true independent implementation
   and rigid ICP from a 1D laser curve is under-constrained. It is now a deprecated alias for the
   constrained IMU/speed prior mode rather than pulling profiles together.
3. **Real IMU stream loading.** The pipeline reads and interpolates `imu.csv` (WXYZ quaternions) at
   each laser-on frame timestamp. Old quaternion-only `gt/poses.npz` files are no longer mistaken
   for ground-truth camera poses.
4. **Laser-plane normalization.** Normal and offset are now normalized together exactly once.
   Previously, a non-unit normal could be normalized twice while `d` was divided only once,
   changing the physical plane. The default laser plane sign now matches the simulator's laser
   apex below the camera in +camera-Y.
5. **Camera distortion.** Camera unprojection/projection now use OpenCV distortion-aware routines
   when non-zero coefficients are supplied. Triangulation calls the common camera model and checks
   frame dimensions rather than duplicating a pinhole-only formula.
6. **Calibration/sensor-size consistency.** The nominal camera config now matches the supplied
   session's 1600x1300 image size and principal point. A session's own calibration is used; the
   pipeline no longer silently assumes a different hard-coded sensor calibration.
7. **Fusion units and normal orientation.** TSDF voxel/truncation values are converted from the
   project millimetre convention to metres for Open3D. Point-cloud normals are oriented toward the
   nearest recorded camera position instead of a hard-coded world coordinate.
8. **BPA meshing.** Ball-pivoting radii are adapted more conservatively from point spacing, and mesh
   cleanup retains significant disconnected surface patches instead of always keeping only the
   largest component. This avoids deleting useful scan patches but cannot create unseen tooth walls.
9. **Evaluation alignment semantics.** `run_eval.py --no-icp` now compares original world-space
   coordinates without silently applying a centroid translation.
10. **Hardware capture format.** The logger writes the shared session schema, a real `imu.csv`, and
    calibration metadata. It waits for the first valid MCU timestamp before recording so it does
    not mix host and MCU clocks. `run_capture.py --calibration` accepts the actual intrinsics and
    laser plane. Nominal default calibration is marked in metadata and warned about by the pipeline.

## Validation performed in this environment

- `python -m compileall -q src scripts tests`: passed.
- Focused tests for smoke/config loading, laser triangulation, session round-trip, curved-path pose
  integration, plane normalization, distortion-aware projection, and trajectory generation:
  17 passed.
- The archived session's measured ridge line was checked against its stored simulator profile;
  extraction was sub-pixel accurate and triangulated points agreed within a few micrometres.
- The corrected IMU/speed integration was replayed against archived ground-truth poses; median
  position error was approximately 0.0008 mm and maximum approximately 0.0013 mm.

A full Open3D end-to-end rerun was not possible in the container used for this patch because
Open3D is not installed in that runtime. Consequently, the two Open3D-dependent test modules could
not be collected here; the user's project environment declares Open3D as a required dependency.
The `WGL` messages from `view_results.py` are Windows OpenGL viewer/context warnings; they are
separate from the geometric errors in the saved mesh and do not fix or explain those errors.

## Remaining physical limitations

- The supplied recording has 200 frames and a camera path of about 19.9 mm; its trajectory plot
  shows a local segment on the right side of the arch. The old session metadata did not preserve
  the CLI frame cap. Regenerate without `--max-frames` for a full-length single occlusal pass.
  Buccal/lingual walls still are not observed, so that pass remains an open surface, not a complete
  360-degree tooth.
- A real IMU provides orientation but not translation. `imu_hybrid` uses a nominal speed prior;
  accurate dental measurements require measured translation (encoder/robust odometry) and a
  calibrated IMU-to-camera mounting rotation.
- The nominal camera calibration is explicitly identified as 8 mm-lens geometry. A different lens
  (including a 25 mm lens) needs fresh checkerboard calibration; do not use nominal intrinsics for
  measurement.


# MeshSegNet integration and validation (current revision)

## Integration changes

- Added a bundled MeshSegNet-compatible network whose parameter names and tensor shapes match the
  public 15-class reference network, so the supplied Man/Max checkpoints do not require a separate
  repository checkout. State dictionaries load strictly; missing or unexpected parameter keys are
  reported as errors instead of silently continuing with random weights.
- Added jaw-aware checkpoint selection from session metadata (`lower` / `upper`) and the `Man` /
  `Max` filename tokens. The two supplied `.zip` filenames are supported, as are PyTorch checkpoint
  files directly and ZIP containers holding one `.tar`, `.pth`, `.pt`, or `.ckpt` checkpoint.
- Matched the reference 15-channel preprocessing: normalized triangle vertices, normalized face
  barycentres, then normalized face normals. The mesh centre uses the equal-weight vertex mean,
  matching Vedo/VTK `center_of_mass()` semantics.
- Builds the two dense adjacency matrices from CPU-side radius-neighbour searches rather than
  allocating a temporary full pairwise distance tensor on the GPU. The inference input is capped
  at 10,000 faces, consistent with the reference model's memory-intensive dense graph input.
- Projects decimated predictions back onto every face of the original reconstructed mesh and writes
  a full colour-labelled mesh, combined teeth-only meshes, separate class meshes, point-cloud labels,
  and face-label/confidence arrays. CUDA out-of-memory errors trigger a CPU retry.
- `run_pipeline.py --segment` now checks PyTorch availability, jaw metadata, and checkpoint selection
  before doing the expensive reconstruction. `view_results.py --help` can run without Open3D.

## Current validation

- `python -m compileall -q src scripts tests`: passed.
- `pytest -q`: 35 passed, 5 skipped in the packaged review environment. The skipped cases need
  Open3D or the external Teeth3DS dataset.
- MeshSegNet forward-pass, probability normalization, 15-channel feature construction, adjacency
  normalization, jaw selection, strict state-dict loading, and nested-checkpoint ZIP loading were
  tested with synthetic data/checkpoints.
- `run_pipeline.py --help`, `run_segment.py --help`, and `view_results.py --help`: passed.

The supplied model archives and Open3D are not part of the uploaded project or this review runtime,
so a real checkpoint inference on a dental scan and the complete Open3D reconstruction-to-export run
remain unverified here. Copy the two original model archives into the configured Windows models
folder, install the project dependencies, and run the README commands to validate the real weights
and capture on the target machine.

## Hardware synchronization audit update

The supplied ESP32 sketch's CSV fields match the host serial protocol. The host capture reader has now
been changed from latest-value sampling to FIFO packet consumption, uint32 `micros()` rollover
extension, sequence-gap detection (when the updated seven-field sketch is used), and queue-overrun
failure. The bundled firmware now refuses to emit scan packets when BNO086 initialization fails or
when camera STROBE is absent/stale; it no longer manufactures free-running fallback frames. This
avoids silent use of default identity orientation and unsynchronized pseudo-frames.

These changes improve software-side safeguards but do not substitute for checking camera/ESP32 frame
alignment on the actual USB camera, nor do they add measured translational tracking. `imu_hybrid`
still integrates nominal speed along the camera scan axis, and IMU-to-camera extrinsics still need
calibration. OpenCV UVC frame timing, camera exposure controls, the ESP32 firmware build/upload, and
physical laser/strobe behaviour remain unverified without the target hardware.
