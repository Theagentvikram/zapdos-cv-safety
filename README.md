# CCTV safety events

## Part 0: cold open

**Q1 - frame budget.** The detector supports 8 / 0.025 = 320 frames/s, or **3.2 FPS per camera** for
100 cameras, before overhead. The decoder supports 600 / 100 = 6 FPS per camera, so inference binds
first once decoding is limited. Decoding all native streams would require 2,500 FPS and also fail.
At 1.4 m/s, a person travels **0.438 m per sampled frame**. Assuming a 2 m zone, crossing takes
1.43 s, or roughly 4-5 samples: sufficient for a basic entry rule, not a very narrow danger strip.
A 2.5 s helmet window has about 8 samples, whose usefulness depends on visible head pixels and
correlated errors. Reduce streams/substream resolution and sample before expensive retrieval;
then measure batching and head recall. A 10-observation rule in 2.5 s must be retuned at 3.2 FPS.

**Q2 - false alarms.** 40 x 3 x 5 x 0.001 x 3,600 = **2,160 false no-helmet person-frames/hour**,
or 17,280 per shift. An 8-of-10 vote only gives the independent-trial reduction if errors are
independent; a hood or missed white helmet can remain wrong across every frame. Two alerts per
shift across 40 cameras means **0.00625 false alerts/camera-hour**. Measure event-level errors on
representative shifts, improve head evidence, abstain on poor views, deduplicate episodes, use
cooldowns, and calibrate cameras using reviewed labels. Zero alerts alone proves nothing.

**Q3 - zone geometry.** Test a ground-contact estimate, not whole-box overlap. This implementation
uses bottom-centre as an explicitly imperfect proxy. Known feet occlusion and bottom truncation
produce unknown; no new entry is inferred from lost-track predictions. Entry and exit require
0.6 s of evidence beyond a +/-5 px boundary band; unknown expires after 1 s. Pose feet or a calibrated
floor homography would improve occluded/angled views. Machinery occlusions need camera masks here.

## Run

Python 3.12, with `pip install -r requirements.txt`. The existing SH17 checkpoint is
`models/sh17/yolo8s.pt`; the default exported graph is `models/sh17/yolo8s_960.onnx`. These weights
were already supplied in the project. Verify their release provenance/licence before publishing;
this change does not download or retrain a model. SteelBench footage remains under its source
licence. Dependencies are pinned because the code uses Ultralytics ByteTrack internals.

```powershell
python scripts/run_ppe_pipeline.py --clips YOUR_CLIP.mp4 --out out
python scripts/run_ppe_pipeline.py --clips YOUR_CLIP.mp4 --zones zones/YOUR_CAMERA.json --camera-id YOUR_CAMERA --out out
python scripts/run_ppe_pipeline.py --clips YOUR_CLIP.mp4 --target-fps 0 --out out_full_rate
python scripts/run_ppe_pipeline.py --clips YOUR_CLIP.mp4 --realtime --out out_paced
python -m unittest discover -s tests -v
```

Default: ONNX CPU, 960px input, 5 detector samples/video-second, original timestamps. `--target-fps 0`
processes every frame. Smaller inputs are experiments, not proven accuracy improvements. An unknown
camera has PPE absence alerts disabled until validated. Use `--allow-unvalidated-absence-alerts`
only for a diagnostic comparison; every such alert records that validation was bypassed.

Config holds detection confidence/size gates, association geometry, tracker settings, temporal
windows and cooldown. Four zone files are **illustrative assessment policies drawn on the clips**,
not actual factory safety restrictions. See [restricted-zone setup](docs/RESTRICTED_ZONES.md) for the
per-camera workflow and single-command example. The tippler camera has an explicit electrical-panel
exclusion; recheck it after camera movement. No motion filter rejects stationary workers.

Output: `<out>/<scene>/<clip-stem>/` (long names are shortened with stable hashes; see `run_summary.json`) contains `events.jsonl` (emitted alerts only), `diagnostics.jsonl`
(including suppressed candidates), annotated video, worker/candidate logs, timelines, snapshots,
and annotated event clips with up to 3 s before/after the alert (trimmed at video ends). Event clips
are 960px wide to bound pre-roll memory. Live records have `end_ts: null`; the completed file has
episode end times and `open_at_end`. Confidence is person-detection evidence, not a calibrated
probability of a safety violation. Reuse an output directory only when intentionally replacing a run.

## Evaluation and results

The latest fixes, 78-test verification, detector comparisons and paced-run limitations are in
[fixes and validation](docs/FIXES_AND_VALIDATION.md). The alternative person and head detectors
were not promoted after broader regressions. `--person-model PATH` is an optional experiment.
`--realtime` simulates paced file input and reports queue-inclusive frame age; it is not an RTSP benchmark.
The [visual reference](reference/README.md) is explicitly agent-reviewed, not human ground truth.

Measured CPU: **Intel Core Ultra 7 256V, 8 cores / 8 logical processors**, Windows 11, CPU execution.
The captured baseline processed 1,300 frames / 52 s of source video in **192.8 s (6.74 FPS)**.
See `docs/RESULTS.md` for the final measured comparison and known failures. Generated datasets
and run outputs are intentionally excluded from this final-code folder. Throughput and video coverage are reported
separately; warm-up/export are excluded. No event precision/recall is claimed without reviewed labels.

```powershell
python scripts/prepare_labels.py --clips YOUR_CLIP.mp4 --out labels
# Review the original videos, fill labels.json and the run-specific identity_map.json.
python scripts/evaluate_events.py --labels labels/labels.json --mapping labels/identity_map.json --predictions out --out metrics.json
```

The evaluator refuses unreviewed templates. It matches person, camera, clip, violation, zone and
alert time (default tolerance 1 s), uses one-to-one maximum matching, includes duplicates in the
precision denominator, and lists false alerts and misses. Mapping is reviewed per run, since
tracker IDs are not ground-truth identities; multiple fragments may map to the same person.

**Honesty / outstanding assessment work:** SH17 misses visible white helmets and can confuse shirts
with vests. Raising PPE thresholds reduces weak positive evidence but is not proof of improved
precision. PPE absence stays disabled on these unvalidated cameras. Floor points can be wrong when
machinery hides feet. Sampling may miss brief events. Person counts are not IDF1 or event recall.
Human-reviewed event labels, required CMOT 0011/0013/0039 clips, a full failure review, and the demo
recording remain to be completed. Record the candidate's actual total time before submission.

Tools: SH17 YOLOv8s, Ultralytics ByteTrack, ONNX Runtime/PyTorch, OpenCV, NumPy, PyYAML;
Codex assisted with code review, implementation, tests, measurements and documentation. No hosted
vision model is used for detection. See `DESIGN_NOTE.md` and `docs/RESULTS.md`.
