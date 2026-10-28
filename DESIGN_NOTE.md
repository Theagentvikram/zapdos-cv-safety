# Design note

**Tradeoffs.** Keep SH17 YOLOv8s at 960px for small heads; 640px is an optional measured experiment,
not the default without event labels. Sample 5 FPS instead of processing all 25 FPS. On 52 s of
local source video that reduces forward passes from 1,300 to 260 (80%). The original full-rate
baseline took 192.8 s; current measurements are in `docs/RESULTS.md`. A 2.5 s absence
window needs 10 eligible negatives and 60% eligible observations; a gap over 1 s restarts evidence.
This delays alerts by at least 2.5 s after evidence begins. Two PPE positives within 0.8 s establish
presence, held for 1 s. One alert per episode, with 30 s cooldown; a continuing pending episode
can alert when cooldown expires. Lost frames never create violations.

**Tracking and trust.** ByteTrack retains the low-confidence pool for second-stage recovery.
Three observations in 0.6 s, median confidence >=0.25 and height >=40px confirm an identity;
border-truncated boxes cannot create one. Motion prediction keeps the jump guard meaningful at
sampled rates. Stitching is time/size/IoU gated and mutually unique; PPE evidence resets on a stitch.
Only recently visible identities are searched. PPE matches all nonduplicate persons, including
untracked ones. Low-confidence/implausible PPE blocks absence rather than becoming a positive.
The known tippler panel is explicitly excluded by camera geometry; this is a local fix with a
documented coverage tradeoff, not a generic machinery classifier.

**Failures.** Workshop and ladle workers wear white helmets missed by SH17; strict mode abstains
until reviewed camera validation. A striped workshop shirt can produce false vest evidence; a
head/torso classifier trained on reviewed crops is the next model experiment. Seated clusters and
glare can hide workers; do not equate fewer IDs with correct tracking. The tippler sampling test
exposed ID splits during walking, motivating motion prediction and a regression test. Known feet
occlusion yields unknown; bottom-centre can still be wrong with unmodelled machinery. Zones use
0.6 s dwell and a +/-5px dead band. Demonstration zones are not actual plant policy. Full low-light,
IR and unseen-clip evaluation is pending, as are CMOT coverage and human event labels.

**100 cameras.** The baseline 6.74 detector FPS permits at most floor(6.74 / 5) = one 5-FPS stream
on this CPU before headroom, even though it handles only 0.27 native 25-FPS streams. Do not claim
100 cameras per laptop. A 100-camera 5-FPS site requires 500 detector FPS plus decoding; benchmark
batched GPU processing and hardware substream decoding to choose boxes. Edge keeps decode, tracking,
rules and bounded evidence buffers; upstream receives events, feedback and selected evidence.
For assumed 0.3 MB snapshot + 6 s at 1 Mb/s, one event is ~1.05 MB. At 30 events/camera/day across
100 cameras, ~3.15 GB/day leaves site (~0.29 Mb/s average), with higher burst needs. These are
planning assumptions, not measured encoder/network results. A production service needs a durable
local outbox, retry/idempotent upload and disk quota for uplink loss; this file runner is not that
service. Onboarding uses a camera ID, normalized zones, masks and reviewed thresholds, no new code.

**Feedback.** Store supervisor confirm/dismiss alongside event ID and reason. Review a random sample
of no-alert footage as well as alerts to find misses. Use camera/shift-held-out labels for threshold
selection and head-crop hard negatives; remeasure duplicates, precision, recall and p95 delay before
promoting any detector or sampling change. Current alert confidence is not a violation probability.
