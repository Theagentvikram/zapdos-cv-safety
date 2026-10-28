"""Evaluate emitted events against reviewed labels and a reviewed per-run person mapping."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ppe_pipeline.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--predictions", required=True, help="events.jsonl or run output directory")
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.predictions)
    files = sorted(path.rglob("events.jsonl")) if path.is_dir() else [path]
    if not files:
        parser.error("no event files found; verify the run output path")
    predictions = [json.loads(line) for file in files for line in file.read_text().splitlines() if line.strip()]
    result = evaluate(json.loads(Path(args.labels).read_text()), predictions,
                      json.loads(Path(args.mapping).read_text()), args.tolerance)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
