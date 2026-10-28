# Restricted zones: setup and one-command runs

## Who defines the zone?

A restricted zone is a site safety policy, not an object category the detector can infer. Choose the actual restricted floor area with the operator/supervisor. For the assessment, draw an illustrative zone on each standard camera view and label its purpose. The current pipeline loads polygons; it does not automatically generate them.

## Configure once per camera

1. Open a representative video frame and locate the floor area to restrict.
2. Select at least three polygon corners around that floor area, in clockwise or counterclockwise order. Do not draw around the complete machine silhouette.
3. Save a JSON file such as zones/camera_01.json. Pixel coordinates are supported with coordinates=pixels. For normalized coordinates, divide pixel x by frame width and pixel y by frame height.
4. Run once and inspect the polygon overlay, worker floor points, and event evidence. Review frames with feet hidden by machines; add known feet-occlusion masks to camera configuration where needed.
5. Reuse the file on later videos from the same unchanged camera. Redraw/revalidate after camera movement, zoom, cropping, or a changed safety policy. Normalized coordinates accommodate image resizing, not a changed viewpoint.

Example format only; these coordinates are not a real safety boundary:

```json
{
  "coordinates": "normalized",
  "zones": [
    {
      "id": "machine_keep_clear",
      "polygon": [[0.60, 0.55], [0.85, 0.55], [0.90, 0.90], [0.55, 0.90]]
    }
  ]
}
```

## One command

```powershell
python scripts/run_ppe_pipeline.py --clips "C:\path\video.mp4" --zones zones/camera_01.json --camera-id camera_01 --out out
```

This single command loads weights/configuration/zones, detects and tracks workers, applies rules, and writes video, alerts, snapshots and evidence clips. Zone setup is camera configuration, not a second inference step. The assessment explicitly asks for committed per-camera zone files and gives a one-command example with a --zones argument.

For videos from configured cameras, the runner matches the filename prefix in configs/ppe_pipeline.yaml and loads that camera's saved zone file automatically. An explicit --camera-id loads the saved camera entry by ID, even when the filename has changed. An unknown ID starts with no saved zones and PPE absence validation disabled; supply --zones for its reviewed policy.

A registered entry uses a path relative to the config directory:

```yaml
cameras:
  - id: camera_01
    match: "camera_01_"
    scene: camera_01
    zones: ../zones/camera_01.json
    helmet_absence_validated: false
    vest_absence_validated: false
```

## Behaviour and limits

- No zone file: person/PPE inference still runs, but there is no zone-entry evaluation or alert.
- A missing or malformed explicitly supplied file fails rather than inventing a zone.
- The rule uses bounding-box bottom centre as a floor-contact proxy, not whole-box overlap.
- Entry and exit each require 0.6 s beyond a +/-5px boundary band; unknown observations expire after 1 s. A worker already inside when the clip begins can alert after confirmation and entry dwell.
- Bottom truncation, another person's box covering the floor point, and configured feet-occlusion masks produce unknown. The code cannot automatically detect all machinery occlusion; bottom-centre may be wrong in those cases.
- Tiny or short/wide boxes also produce unknown (default minimum height 80px, height/width 1.6). These conservative gates can suppress crouched or partly hidden workers. The worker CSV records floor-point visibility and per-zone states for review.
- PPE absence validation is a separate setting. Supplying zones does not validate a camera's helmet or vest detector.
- Automatic scene segmentation could suggest polygons in a future setup tool, but an operator still needs to approve which areas are actually restricted.
