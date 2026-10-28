# Visual diagnostic reference

`visual_reference.json` is an agent-reviewed, nine-frame / three-camera diagnostic with 22 person boxes. It explicitly records `human_reviewed: false`. Unknown PPE is excluded from known-label accuracy, and unknown predictions count as abstentions. It cannot qualify cameras or establish event precision/recall.

Original frame PNGs are retained outside the final code folder at:
`C:\Users\chira\OneDrive\Desktop\ZAPDOS_video_tests_20261027\review_v2\reference`.
Source hashes and frame numbers are in the JSON. Human review may correct these provisional labels; preserve reviewer provenance and the camera split.

From the project directory:

```powershell
python scripts/evaluate_reference.py --reference reference/visual_reference.json --images "C:\Users\chira\OneDrive\Desktop\ZAPDOS_video_tests_20261027\review_v2\reference" --out "C:\path\diagnostic_results"
```

Optional `--person-model PATH` and `--head-model PATH` enable explicit candidate comparisons. Neither candidate is the default or bundled with this final project. See `docs/FIXES_AND_VALIDATION.md` for the observed regressions.
