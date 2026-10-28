"""Report whether reviewed camera evidence meets a conservative alert-admission policy.

Evidence JSON: camera_id, type, reviewed, human_reviewed, reviewer:{kind:human},
complete_exposure_reviewed, exposure_hours, true, false, duplicates, missed.
This command never edits the runtime configuration.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ppe_pipeline.calibration import assess_calibration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--cameras",type=int,default=40)
    p.add_argument("--false-alerts-per-shift",type=float,default=2)
    p.add_argument("--shift-hours",type=float,default=8)
    args=p.parse_args()
    if args.cameras<=0 or args.shift_hours<=0 or args.false_alerts_per_shift<=0:
        p.error("camera count, shift hours and false-alert budget must be positive")
    result=assess_calibration(json.loads(Path(args.evidence).read_text()),
                              args.false_alerts_per_shift/(args.cameras*args.shift_hours))
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
