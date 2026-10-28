"""Conservative admission check for a human-reviewed camera/violation validation run.

Short clips and agent-generated labels cannot establish a site false-alarm budget.
No configuration flags are changed by this module.
"""
import math


def assess_calibration(evidence, max_false_alerts_per_camera_hour=.00625, min_recall=.9,
                       min_labelled_events=20, confidence=.95):
    reasons=[]
    human=(evidence.get("reviewed") is True and evidence.get("human_reviewed") is True and
           evidence.get("reviewer",{}).get("kind")=="human")
    if not human:
        reasons.append("complete human review is required; agent review is diagnostic only")
    if evidence.get("complete_exposure_reviewed") is not True:
        reasons.append("both alert and no-alert footage must be reviewed")
    if evidence.get("type") not in {"no_helmet","no_vest"} or not evidence.get("camera_id"):
        reasons.append("one explicit camera and PPE violation type are required")
    hours=evidence.get("exposure_hours",0)
    counts=[evidence.get(k,0) for k in ("true","false","duplicates","missed")]
    if (not isinstance(hours,(int,float)) or not math.isfinite(hours) or hours<=0 or
            any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in counts)):
        raise ValueError("positive finite exposure and nonnegative integer counts required")
    if not 0<confidence<1 or max_false_alerts_per_camera_hour<=0:
        raise ValueError("invalid confidence or false-alert budget")
    true,false,duplicates,missed=counts
    recall=true/(true+missed) if true+missed else None
    if true+missed<min_labelled_events:
        reasons.append(f"at least {min_labelled_events} labelled violation events required")
    if recall is None or recall<min_recall:
        reasons.append("event recall is below the acceptance threshold or undefined")
    # Exact one-sided Poisson upper bound when no false alerts were observed.
    # Any observed false alert fails this intentionally conservative initial policy.
    upper=-math.log(1-confidence)/hours if false+duplicates==0 else None
    if false+duplicates:
        reasons.append("false or duplicate alerts observed; repair and revalidate")
    elif upper>max_false_alerts_per_camera_hour:
        reasons.append("too little exposure to bound the requested false-alert rate")
    return {"camera_id":evidence.get("camera_id"),"type":evidence.get("type"),
            "eligible":not reasons,"reasons":reasons,"human_reviewed":human,
            "event_recall":recall,"false_alerts":false+duplicates,
            "false_alert_rate_upper_per_hour":upper,"confidence":confidence,
            "required_zero_false_alert_hours":-math.log(1-confidence)/max_false_alerts_per_camera_hour,
            "note":"Poisson bound assumes representative stable exposure; it is not a guarantee after camera/shift changes. No flags were modified."}
