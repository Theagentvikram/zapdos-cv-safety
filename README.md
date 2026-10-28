# CCTV safety events: detect, track, alert

Fixed factory CCTV in; deduplicated `no_helmet`, `no_vest` and `zone_entry` alerts out, each as one
JSON line with an annotated snapshot and a clip. Design note: [`docs/DESIGN.md`](docs/DESIGN.md)
(one A4 page: [`docs/DESIGN.pdf`](docs/DESIGN.pdf)). Every table in detail: [`docs/RESULTS.md`](docs/RESULTS.md).

## Part 0

**Q1. Frames.**
- **Decode binds first.** 100 × 25 fps = 2,500 fps of decode against 600; dropping frames after decode
  saves nothing (H.264 P-frames need their predecessors). Set the cameras or substreams to 5 fps: 500 fps
  fits.
- **Then the detector:** 8 / 25 ms = 320 fps, so **3.2 fps per camera**.
- **Zone entry: enough.** At 1.4 m/s that is **0.44 m per frame**. A 3-frame confirmation (0.9 s) fires
  about 1.3 m past the edge, so draw the polygon with a 1.5 m buffer around the hazard. Measured:
  replayed at 3.2 fps, the dev clips give the same 4 of 7 zone entries as at 10 fps.
- **Missing hard hat: the limit is head size.** 3 s gives 10 frames to vote on, but SteelBench heads
  are **26 px** tall (median) at 1080p and about **9 px** at a 640 input, where SH17 finds 8 of the 33
  helmets it finds at 1280 ([`eval/pixel_budget.py`](eval/pixel_budget.py)). Measured at 3.2 fps: 2 of
  5 helmet clips caught (3 at 10 fps).
- **Change first: sampling and streams** (camera-side 5 fps, free). Then the model input rather than
  more edge boxes: detect people cheaply and judge the helmet on a crop from the native frame (2.7 ms
  per frame).

**Q2. False alarms.**
1. 40 × 3 × 5 = 600 person-frames/s. At 0.1% that is **2,160 false frames per hour**.
2. "8 of 10" assumes independent errors: C(10,8)·0.001⁸ ≈ 4.5×10⁻²³. But errors stick to a person (a
   beanie, a hood, a backlit head), and that person votes 10 of 10. Voting removes errors that vary
   between frames, not those that persist; without dedup a passing vote re-fires every window.
3. **Count alerts per person-episode.** At about 2 min in view, 120 people give 28,800 episodes per
   shift, so 2 false alerts per shift allows at most 1 false alert per 14,400 episodes. What gets
   there: one event per episode (cooldown, ID-switch merging); **"can't tell"** where the head can't
   be judged (no head box, < 18 px, frame edge), where persistent errors occur; in-domain training and
   a per-camera threshold; dismissals fed back as hard negatives (not built here). Here the classifier
   and these gates took compliant clips with a false alert from **16/25 to 0/25**.

**Q3. Where is the person.**
1. **The foot point** (box bottom-centre). The worker 1 m in front overlaps the zone in the image, but
   their feet are outside.
2. **Feet hidden** (box much shorter than the track's recent height): feet = box top + remembered
   height, slowly decaying.
3. **Hysteresis in person heights:** enter at 0.1 × height inside the edge, leave at 0.1 × height
   outside; plus 0.5 s persistence and a 60 s cooldown. Unit-tested in
   [`tests/test_rules.py`](tests/test_rules.py).

## Run

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/download_data.py              # SteelBench sample + SH17 weights; links the labelled clips
python scripts/export_models.py --detector   # macOS only: CoreML fp16 detector
python run.py --video clips/sb_crm_workshop_0109.mp4 --zones zones/sb_crm_workshop_0109.json --out out/
```

- **Options:** `--zones` is optional. Without CoreML (Windows, Linux) the detector runs in PyTorch on
  CUDA or the CPU. `--show` plays it live; `--save-video` writes `annotated.mp4` (red violating, green
  OK, grey can't tell). Several cameras in one process: `--video a.mp4 b.mp4 --zones a.json b.json`.
- **Output** in `out/<camera_id>/` (sample committed): `events.jsonl`, one line per alert as it fires
  (IDs, type, zone, times and delay, confidence, box, media paths, merged track IDs; `end_ts` final at
  the end), `snapshots/`, `clips/` (3 s before to 3 s after), `perf.json`.
- **Config:** rule numbers in [`configs/rules.yaml`](configs/rules.yaml); the six clips' zone polygons in
  [`zones/`](zones/). New camera: draw its polygons on one frame; `scripts/zone_reference.py` saves that
  view. `./reproduce.sh` reruns everything below.

| Stage | Choice | Why |
|---|---|---|
| Detect | SH17 YOLOv8s (person, head, helmet, vest) at 736×1280, CoreML fp16 | heads are 26 px; at 640 SH17 keeps 8 of 33 helmets |
| Track | **ByteTrack** on persons, 3 s buffer | crops too small for appearance ReID; its low-score pass keeps part-hidden people |
| Helmet | MobileNetV3-small on a 64 px head crop from the native frame | SH17's helmet boxes false-alarm on 16/25 compliant clips |
| Can't tell | no head box, head < 18 px, frame edge; vest: person < 100 px | persistent errors occur there (Q2) |
| Events | PPE ≥ 70% of readable frames in 2 s for 1.5 s; zones ≥ 60% in 1 s for 0.5 s; 60 s cooldown | one alert per person, type and episode, at any fps |
| Dedup | a new track inherits an open event if nested in the owner's box, or where the lost owner was | ID switches don't re-alert |
| Camera moved | once a second, match the frame to the view the zones were drawn on | a bumped camera pauses its zones instead of failing silently |

## Results

**Events on 6 hand-labelled clips** ([`eval/events_eval.py`](eval/events_eval.py)); a repeat alert on
one event is a duplicate and counts against precision. **Dev** ([`labels/`](labels/)): tuned on.
**Held out** ([`labels_heldout/`](labels_heldout/)): 3 other cameras, labelled before the first run.

| Clip | Labelled events | Alerts | True | False | Duplicates | Missed | Median delay (s) |
|---|---|---|---|---|---|---|---|
| sb_crm_workshop_0109 (5 people, occlusion) | 8 | 8¹ | 7 | 0 | 0 | 1 | 1.52 |
| sb_red_lrs1_0007 (far, haze) | 3 | 1 | 1 | 0 | 0 | 2 | 2.22 |
| sb_tippler_lowlight_0062 (night) | 3 | 3 | 3 | 0 | 0 | 0 | 1.72 |
| **Dev total** | **14** | **12** | **11** | **0** | **0** | **3** | |
| sb_bf_cdi_cc2_0031 (hopper grate) | 3 | 3 | 3 | 0 | 0 | 0 | 1.92 |
| sb_rers1_0012 (5 people, coveralls) | 6 | 7 | 5 | 2 | 0 | 1 | 1.52 |
| sb_tar_lowlight_0000 (night, legs only) | 1 | 0 | 0 | 0 | 0 | 1 | – |
| **Held-out total** (re-run) | **10** | **10** | **8** | **2** | **0** | **2** | |

¹ Plus one no_vest alert on a person whose vest can't be judged by eye (not counted). Delay: from the
labelled start.

- **Held out, first run: P 0.73** (8/11, 95% CI 0.43–0.90), **R 0.80** (8/10, CI 0.49–0.94). These
  blind numbers are the best estimate for an unseen camera. Two rules came after it (18 px heads; zone
  confirmation at 0.5), set without the held-out clips but after seeing their errors, so the re-run
  above (P 0.80) is not blind. No duplicates; 9 of 16 people keep one track ID
  ([`eval/track_eval.py`](eval/track_eval.py)).
- **Errors.** False now: two coveralls with reflective tape. Removed by the two rules: a 15 px night
  helmet and a grate phantom. Missed: people hidden or 40–80 px tall in haze, dark hair from behind,
  legs under a platform. Median delay: zone 0.9 s, no_vest 1.8 s, the two no_helmet about 4 s.

**PPE on all 50 SteelBench sample clips** (scored per clip against its per-person labels,
[`eval/steelbench_clips.py`](eval/steelbench_clips.py); no camera used in training): the head-crop
classifier catches **3/5** helmet-violation clips and false-alarms on **0/25** compliant ones, against
4/5 and 16/25 for SH17's own helmet boxes; vests 21/26. The labels have no boxes or times, so an alert
counts as right if its clip has enough violators: precision is an upper bound (helmet ≤ 1.0, SH17
boxes ≤ 0.13, vest ≤ 0.82). Low light is the helmet weak spot. **Camera moved** ([`eval/view_eval.py`](eval/view_eval.py)): no trips on 50 still clips; every
camera swap and every 3% pan or 10% zoom caught.

**Speed.** MacBook Air M2 (2022, Mac14,2): 8-core CPU (4P + 4E), 8-core GPU, 16-core Neural Engine,
8 GB, macOS 27.0.1. 1080p H.264 at 25 fps in, 10 fps processed ([`bench_streams.py`](scripts/bench_streams.py)).

| Hardware | Input res. | Decode ms | Detect ms | Track + rules ms | FPS per stream | Streams at 10 fps |
|---|---|---|---|---|---|---|
| M2 Neural Engine (CoreML fp16) | 1080p → 736×1280 | 2.1 | 21.8 | 3.2 (helmet 2.7) | **36.2** | **4** (measured, 10.5 fps each) |
| M2 **CPU only** (CoreML fp16) | 1080p → 736×1280 | 2.2 | 83.4 | 3.7 (helmet 3.1) | **11.1** | 1 (2 streams: 8.7 fps each) |
| M2 CPU only (PyTorch fp32) | 1080p → 736×1280 | 2.1 | 217.0 | 4.0 (helmet 3.3) | 4.5 | 0 |

- **Targets:** ≥ 10 fps on the accelerator: met 3.6×. ≥ 5 fps on CPU: met 2.2× with `--cpu-only` (all
  94 benchmark detections kept; 18 of 24 labelled events caught vs 19). PyTorch on CPU: 4.5 fps.
- **Several cameras:** 4 separate processes run 10.5 fps each; 6 swap on 8 GB. One process with shared
  models ([`bench_multi.py`](scripts/bench_multi.py)) keeps up with 3 cameras at 10 fps and fits 6 in 2.2 GB.
- **Stretch goals:** CoreML fp16 export (21.7 vs 53.2 ms on the GPU, all 94 detections kept); head
  classifier (+2.7 ms).

## Used, and time

- **Models:** SH17 YOLOv8s (release weights); MobileNetV3-small fine-tuned on Hard Hat Workers plus
  head crops from 16 SteelBench cameras outside the evaluation set
  ([`build_steelbench_heads.py`](scripts/build_steelbench_heads.py), [`train_helmet.py`](scripts/train_helmet.py));
  threshold at 2% false "bare" (5 of 231 helmets) on 2 further calibration cameras.
- **Data:** [SteelBench](https://huggingface.co/datasets/ThinkingHub/PP) (CC BY-NC 4.0; `out/` is cut from it),
  [Hard Hat Workers](https://huggingface.co/datasets/keremberke/hard-hat-detection) (CC BY 4.0),
  [SH17](https://github.com/ahmadmughees/SH17dataset) weights.
- **Libraries:** ultralytics 8.4.191 (YOLO, ByteTrack), PyTorch 2.14, coremltools 9, ONNX Runtime 1.31,
  OpenCV 5.
- **AI:** Claude Code for code, sweeps and drafts; labels, zones and failure analysis checked by eye.
- **Demo video:** _TODO: link_. **Time spent:** _TODO before submitting: hours actually spent_

## Honesty: where it is most likely wrong

- **The labelled set is tiny** (24 events, one labeller): P 1.00 tuned on, 0.73 held out; the 50 clips
  are the better guide.
- **Helmet recall** is 3/5 clips (CI 23–88%). The threshold comes from 231 helmets on 2 cameras;
  night, IR or a new site will move it, so calibrate per camera for one shift before going live.
- **Rarely or never detected:** people 40–80 px tall in haze, or mostly hidden: 4 of 11 zone events
  missed.
- **Data quirks:** face-blur patches are in the training crops; SteelBench counts blue coveralls as a
  vest, and so do the vest numbers. **Speed** is from one M2; no CUDA GPU measured.
- **How to tell in production:** per-camera dismiss rate (precision); alert-rate drift and
  `camera_moved` events; a weekly audit of 50 random non-alerted tracks (recall); a rising share of
  "can't tell" readings.
