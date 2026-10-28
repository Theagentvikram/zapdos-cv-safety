# Bug fixes and validation, 27 October 2026

The work follows the requested priority order. Code fixes and diagnostic comparisons are complete. Human event annotation and camera qualification remain outstanding; neither is inferred from a model's output.

## 1. Episode continuity and output correctness

PPE episodes now survive a brief dip below the rolling negative-evidence threshold. Previously that dip could end an ongoing violation and create another alert for the same episode. Confirmed PPE recovery, stale evidence or an identity reset closes the episode. A gap between callbacks also expires stale evidence. An ineligible observation cannot start a new episode, and cooldown retries require current eligible evidence.

Zone entries have episode counters. Event IDs distinguish episodes; decision IDs distinguish emitted/suppressed decisions and name snapshots, preventing snapshot replacement. Long camera and clip names are shortened with a stable hash to avoid the Windows path failure encountered in testing. The actual output paths are recorded in `run_summary.json`.

Explicit camera selection now loads its saved policy even with a renamed input file. Unknown camera IDs cannot inherit another camera's validation or zones through the input filename.

## 2. Independent review evidence

`reference/visual_reference.json` contains nine independently inspected source frames from three cameras, with 22 person boxes and visible PPE labels. BF ground and tippler are development cameras; bagging is a separate test camera. Ambiguous PPE is labelled unknown. This is agent review, explicitly **not human review**, and sparse frames cannot establish event recall, event timing or tracking accuracy.

`prepare_reference.py` extracts source frames with source hashes; `evaluate_reference.py` measures person precision/recall and PPE coverage, errors and abstentions. Missing people and unknown PPE predictions do not become correct negatives. Camera overlap between development and test splits is rejected. `prepare_labels.py` remains the workflow for full-video event annotation and per-run identity mapping. Its unreviewed templates must be completed against the original videos, including no-alert intervals, before event evaluation.

## 3. Detector and crop comparisons

| Diagnostic configuration | Development person TP / FP / FN | Bagging test person TP / FP / FN |
|---|---:|---:|
| Default SH17 960 | 6 / 4 / 0 | 0 / 1 / 16 |
| Independent COCO YOLOv8n person branch | 6 / 0 / 0 | 5 / 0 / 11 |

These are small, agent-labelled frame diagnostics, not general accuracy estimates. The independent branch helped these frames but a broader full-video regression failed to confirm the occluded BF conveyor worker. It remains opt-in via `--person-model PATH`; its experimental weights are outside the final project. It retains SH17 PPE detections and replaces only the person branch.

A head-crop hardhat/no-hardhat candidate recovered two known positive helmets in the test frames, but falsely called a cloth covering a hardhat on one development frame. It was rejected for the runtime default. The optional crop comparison abstains on tiny, conflicting or unlocalised evidence. A 640px PPE experiment also lost development helmet evidence and was rejected. Default remains the SH17 960 graph. None of these comparisons resolves the bagging camera's missed people and PPE.

Candidate provenance: official Ultralytics YOLOv8n assets, and the pinned `keremberke/yolov8n-hard-hat-detection` Hugging Face checkpoint at revision `287bafa2feb311ee45d21f9e9b33315ff6ff955d`. Experimental models were not bundled or promoted. ONNX paired sessions disable thread spinning to reduce contention; no general speedup is claimed.

## 4. Tracking and floor evidence

Identity stitching now has a small torso-colour appearance veto when sufficiently large crops exist. Incompatible appearances prevent a geometry-only stitch; missing appearance evidence falls back to the existing conservative geometry checks. This is not re-identification and has no measured IDF1 gain yet.

Floor-point inference additionally abstains for tiny, short or implausibly wide person boxes. Defaults require height at least 80 pixels and height/width at least 1.6. This reduces unsupported entries from partial-body boxes but can also suppress valid crouched/occluded workers. It does not recover hidden feet. CSV output records floor-point coordinates, visibility and zone state so abstentions are inspectable. Per-camera reviewed polygons remain required; restrictions are not inferred from video.

## 5. Calibration and throughput

`check_calibration.py` evaluates complete human-reviewed exposure, event recall, false alerts and duplicates without changing any runtime flag. Under the assessment budget of two false alerts per eight-hour shift across 40 cameras, zero observed false alerts requires about 479.3 representative camera-hours for a one-sided 95% Poisson upper bound. This is an initial admission policy with explicit assumptions, not a guarantee. Short clips and agent labels cannot qualify a camera. PPE absence validation remains disabled.

`--realtime` paces recorded frames and measures age from their scheduled arrival through the local input queue to main-thread completion. It is a paced-file simulation, excluding camera/network capture and output delivery. It does not implement a live RTSP latest-frame/drop policy.

The new 15-second BF conveyor paced run processed 75 sampled frames in 217.51 seconds (0.34 processed FPS), with median frame age 26.97 seconds and p95 201.46 seconds. Inference median was 752.97 ms; the run also had substantial stalls. The current machine/run did **not** meet 5 FPS or low-latency operation. Earlier optional-person runs ranged from 6.66 FPS on two clips to 2.05 FPS on a broader four-clip run. Hardware load and scheduling were uncontrolled; those observations do not isolate a detector speed improvement. Queue age is now visible instead of being hidden by source timestamps.

## Verification and retained artifacts

78 unit/integration tests passed, including episode recovery/staleness, cooldown eligibility, identity stitching, partial floor boxes, reference matching, camera selection and calibration admission. Output verification opened snapshots and clips for the four-video regression (two emitted zone alerts) and completed the paced run check. Zero alerts in the paced run is not evidence of zero errors.

Raw videos, source-frame PNGs, rejected models, predictions and run summaries remain outside the final code folder in `C:\Users\chira\OneDrive\Desktop\ZAPDOS_video_tests_20261027\review_v2`. Earlier source videos remain in `C:\Users\chira\OneDrive\Desktop\ZAPDOS_nonfinal_backup_20261027`. No new model has been trained. Next accuracy work requires human event review and representative labelled training/validation data, followed by measured recall and ID continuity before promotion.
