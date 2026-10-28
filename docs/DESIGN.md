# Design note

decode → SH17 YOLOv8s (CoreML fp16, 736×1280) → ByteTrack → native-res head crop → helmet classifier →
readings (violating, OK or "can't tell") → vote, persistence, dedup → alerts. Measured on a MacBook Air
M2 (8 GB).

## Tradeoffs

| Choice | Options | Cost | Effect |
|---|---|---|---|
| Detector input | 960² / **736×1280** | 23 / **22** ms | 960² keeps 40/53 persons, 6/12 helmets, 2/6 vests of the 1280 reference; 736×1280 keeps all of them |
| Model size | v8n / **v8s** | 15 / **22** ms | v8n keeps 8/12 helmets and **0/6 vests** |
| Backend | PyTorch MPS / **CoreML fp16** (NE) | 53 / **22** ms | keeps all 94 reference detections; detector alone on CPU: PyTorch 187, CoreML 71 ms |
| Helmet | SH17 box / **classifier** | +2.7 ms | false-alarm clips **16/25 → 0/25**; caught 4/5 → 3/5 |
| Confirmation | 1.0 / **1.5** / 2.0 / 3.0 s | helmet median delay 1.5 / **1.6** / 2.6 / 3.1 s | helmet clips caught 3, **3**, 2, 2 of 5; vest 21, **21**, 20, 15 of 26 |
| Frame rate | **10** / 5 / 3.2 fps | 3.1× / 1.6× / 1× detector | helmet clips 3 / 2 / 2 of 5; vest 21 / 20 / 19 of 26; labelled recall 0.79 / 0.71 / 0.71; night-clip delay 1.7 / 5.3 / 5.2 s |

## Failure modes, from the clips

- **Occlusion (0109).** Two seated men, one behind the other, get one box: one zone entry missed, and
  their vests read "can't tell", not "missing". Lowering the detector and ByteTrack floors from 0.10
  to 0.05 does not recover it (dev unchanged; one more vest alert on 50 clips). Next: count heads in
  the zone.
- **Crowding and ID switches.** Of 16 detected labelled people, 9 keep one ID; of 9 switches, 3 come
  after more than 3 s unseen (new ID by design); none made a duplicate alert. The evaluation caught a
  bug (0109): a walker inherited a neighbour's zone event when the neighbour dropped one frame. Fixed:
  tracks ever seen side by side are two people. Next: a longer track buffer on cameras where people
  pass behind long machines.
- **Low light (0062).** A green helmet 15 px tall read "bare" in every head-box frame for 1.5 s: an
  error that persists across frames, which voting can't fix. Heads under 18 px now read "can't tell"
  (was 10), which removed it; on 43 sample clips from non-held-out cameras, false-alarm clips go
  1/25 → 0/25 with catches kept. IR is untested, so an IR camera runs silent for one shift and is
  calibrated first. On a held-out night camera, a worker showing only legs under a platform was never
  boxed. Next: fine-tune the detector on partly visible people.
- **Camera angle and distance (0007).** Two people 40–80 px tall in haze are tracked in 1% and 0% of
  their labelled frames: two zone entries missed. A zoomed second pass over the zone does not detect
  them either, so the fix is a detector trained on small people. A bumped or re-angled camera slides
  the zones off their floor: a once-a-second match to the view they were drawn on emits
  `camera_moved` and pauses them. On a held-out clip re-angled by 2° and 5%, it fires at 4.0 s;
  without it, one of two zone entries is silently missed. No false trips on 50 still clips.
- **Look-alikes.** A cloth head-wrap and a hood count as "no helmet" (correct); a white cloth cap is
  borderline. A body cut off at the frame edge gave a false no_vest, so edge boxes now read "can't
  tell". Posters or reflections of people: none occurred in these clips.
- **Held out (3 unseen cameras, first run P 0.73).** Coveralls with reflective tape read as no_vest
  (2 false): whether they count is the site's vest rule. A small phantom "person" in a grate's
  texture fired a zone alert; a zone now needs a track seen once at ≥ 0.5 confidence (re-run P 0.80,
  not blind).

## Scaling to 100+ cameras

- **Per box.** One M2 sustains **4 streams at 10 fps** (36 / 24 / 10.5 fps each at 1 / 2 / 4 streams),
  1 CPU-only. The detector is 80% of frame time, so cameras per box follow its throughput: Q1's box
  (320 fps) at 5 fps per camera gives ≈ 60, so **2 boxes per 100 cameras** (4 at 10 fps). Six
  separate processes overflow 8 GB; one process with shared models keeps up with 3 cameras at 10 fps
  and fits 6 in 2.2 GB.
- **On the box:** everything, from decode to clip encoding. **Upstream:** only events (0.4 KB),
  snapshots (0.7 MB) and clips (0.8 MB): about 50 MB per site per day at 30 alerts.
- **Uplink drop (planned, not built):** local alerting continues. Events queue on disk and resend in
  order; `event_id` makes a resend idempotent.
- **New camera:** no code. Draw a zone JSON on one frame and save that view (`zone_reference.py`);
  run one shift silent, then set its threshold to the false-alarm budget in a per-camera copy of
  `rules.yaml` (`--config`).

## Operator trust (planned, not built)

Each dismissal (1) adds the head crop as a per-camera hard negative, the mechanism that moved the
classifier in-domain here with 349 hand-labelled crops; (2) re-fits the camera's threshold to 2 false
alerts per shift, as the shipped one was set on 2 calibration cameras; (3) after repeats at one spot,
proposes a mask (posters, mannequins). Retraining touches only the 6 MB classifier.
A camera's dismiss rate is its live precision; over budget, its threshold rises and it is flagged for
review.
