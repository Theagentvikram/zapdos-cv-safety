# Final validation summary

**Latest follow-up:** [fixes and validation](FIXES_AND_VALIDATION.md) records the current bug fixes,
78 passing tests, rejected detector candidates and a paced run that fell below 5 FPS.
The numbers below are historical single-run measurements, not a sustained throughput guarantee.

Measured 2026-10-27 on Intel Core Ultra 7 256V (8 cores / 8 logical processors), Windows 11, ONNX Runtime CPU. Four 1080p / 25 FPS SteelBench clips total 52 seconds. Wall time includes output writing; export and warm-up are excluded. These are single runs on a shared laptop.

| Configuration | Detector calls | Wall seconds | Processed FPS | Source coverage |
|---|---:|---:|---:|---:|
| Original baseline | 1300 | 192.78 | 6.74 | 0.270x |
| Final default (960px, 5 FPS sampling) | 260 | 38.10 | 6.82 | 1.365x |
| Final code, every source frame | 1300 | 209.73 | 6.20 | 0.248x |

The final default uses 80% fewer detector calls. Its roughly 5x faster video coverage comes mainly from sampling, not an equal-work inference speedup. The 640px experiment was rejected as default because it confirmed no workers in the ladle clip.

The final four-clip run produced five demonstration-zone alerts with valid records, readable snapshots and playable clips. All 55 unit/integration tests passed. The sample zones are assessment policies, not actual factory restrictions. Event-level precision/recall remains unmeasured without reviewed labels.

Unseen-video checks: HME workshop (10 seconds) processed in 6.47 seconds; BF CDI Ground CC-2 (15 seconds) processed in 9.60 seconds. Both exposed unresolved PPE or false-person errors. In BF CDI, the real worker kept one ID while a small background feature received two additional IDs. Diagnostic mode raised suspected no-vest/no-helmet alerts, with camera validation explicitly bypassed.

Outstanding assessment evidence: reviewed event labels, CMOT 0011/0013/0039, and demo recording. CMOT's official Alipan host required sign-in. General false detections and white-helmet misses remain unresolved; default PPE absence alerts stay disabled on unvalidated cameras.

Generated videos, datasets, caches, experimental models and historical run reports were removed during final-code cleanup at the user's request. This document retains the measured summary; raw run artifacts are not included. New measurements are produced by the runner and can be checked using scripts/verify_outputs.py.
