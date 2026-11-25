# Camera ROI calibration

The dashboard's **Setup → Camera ROI** editor lets the operator define the optical opening without editing code.

## Draw and save

1. Connect the physical Arducam and click **Check camera**.
2. Open **Setup → Camera ROI**. The editor displays the live 1600 × 1200 camera image.
3. Click around the inside edge of the usable opening, placing vertices in order. The polygon closes visually after three points.
4. Drag the circular handles to refine the outline. **Undo point** removes the last vertex; **Clear** removes all vertices from the editor.
5. Enable **Apply ROI to scan** and click **Save ROI**. The enabled area is shown in green and the outside area is shaded.

The coordinates are normalized to the image dimensions and stored in `configs/roi.json`. The server validates the polygon before saving. Saving is disabled while a scan is capturing or processing.

## How it affects reconstruction

The dashboard snapshots the current ROI to `data/sessions/<session-name>/roi.json` when a scan starts. This makes each scan reproducible even if the operator edits the project ROI later. During both live reconstruction preview and final session reconstruction, ambient-subtracted laser images are masked before ridge extraction. Only laser ridge pixels inside the polygon can be triangulated.

Uncheck **Apply ROI to scan** and save to disable the mask and process the full frame. Existing sessions without `roi.json` also process the full frame.

## Important limitation

The ROI removes image regions outside the selected polygon; it does not identify all chassis surfaces that happen to lie inside that polygon, and it cannot recover tooth surfaces physically occluded by the chassis. Set the polygon slightly inside the optical opening boundary to avoid reflections from the housing. Verify it with a stationary calibration object and inspect the resulting point cloud before a full scan.
