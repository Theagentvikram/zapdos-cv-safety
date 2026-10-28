"""Alert logic on synthetic tracks: one person = one alert, flicker doesn't alert, ID switches and
re-detections don't double-alert, and zone membership uses the feet with hysteresis.

python -m pytest tests -q
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.events import EventEngine, Observation  # noqa: E402
from zapdos.zones import ZoneSet  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/rules.yaml").read_text())
FPS = 10


def feed(engine, frames, fps=FPS, key=("no_helmet", None)):
    """frames: list of [(track_id, box, value)] per processed frame; returns all alerts."""
    alerts = []
    for i, people in enumerate(frames):
        obs = [Observation(tid, box, {key: (value, 0.9)}) for tid, box, value in people]
        alerts += engine.step(i, i / fps, obs)
    return alerts


def box(x, y=400, w=60, h=180):
    return [x, y, x + w, y + h]


def test_sustained_violation_alerts_once_after_min_duration():
    alerts = feed(EventEngine(CFG["events"], "cam", FPS), [[(1, box(100), True)]] * 100)
    assert len(alerts) == 1
    assert abs(alerts[0].alert_ts - CFG["events"]["default"]["min_duration_s"]) < 0.25


def test_rules_keep_their_meaning_at_lower_frame_rates():
    # 3.2 fps is the per-camera frame budget from README, Part 0 (Q1): a 1 s zone window holds 3
    # frames, so a frame-count minimum of 4 could never be met
    for fps in (10, 5, 3.2):
        for key in (("no_helmet", None), ("zone_entry", "z")):
            p = {**CFG["events"]["default"], **CFG["events"].get(key[0], {})}
            frames = [[(1, box(100), True)]] * int(10 * fps)
            alerts = feed(EventEngine(CFG["events"], "cam", fps), frames, fps, key)
            assert len(alerts) == 1, (fps, key)
            assert alerts[0].alert_ts - p["min_duration_s"] < 1 / fps + 1e-6, (fps, key)


def test_flicker_does_not_alert():
    frames = [[(1, box(100), i % 3 == 0)] for i in range(100)]  # violating 1 frame in 3
    assert feed(EventEngine(CFG["events"], "cam", FPS), frames) == []


def test_unknown_frames_never_count_as_violations():
    assert feed(EventEngine(CFG["events"], "cam", FPS), [[(1, box(100), None)]] * 100) == []


def test_id_switch_while_walking_inherits_the_event():
    frames = [[(1, box(100 + 3 * i), True)] for i in range(40)]
    frames += [[] for _ in range(10)]                                   # lost for 1 s
    frames += [[(7, box(250 + 3 * i), True)] for i in range(40)]        # new id, next to where #1 was
    engine = EventEngine(CFG["events"], "cam", FPS)
    assert len(feed(engine, frames)) == 1
    assert engine.events[0].merged_track_ids == [7]


def test_reacquired_after_long_occlusion_in_the_same_place():
    frames = [[(1, box(100), True)]] * 40 + [[]] * 80 + [[(9, box(102), True)]] * 40   # 8 s behind a pillar
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 1


def test_second_box_on_part_of_the_same_body_does_not_alert():
    upper_body = [105, 400, 155, 480]
    frames = [[(1, box(100), True)]] * 30 + [[(1, box(100), True), (2, upper_body, True)]] * 40
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 1


def test_neighbour_does_not_inherit_when_the_owner_drops_a_frame():
    # #1 is violating; #2 walks up beside them already on screen; #1's detection flickers out for two
    # frames just as #2's own violation confirms: #2 is a second person, not #1 re-identified
    frames = [[(1, box(100), True), (2, box(400 - 2 * i), i >= 20)] for i in range(40)]
    frames[35] = frames[36] = [(2, box(330), True)]
    frames += [[(1, box(100), True), (2, box(320), True)]] * 20
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 2


def test_two_people_side_by_side_alert_separately():
    frames = [[(1, box(100), True), (2, box(170), True)]] * 40
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 2


def test_cooldown_stops_realert_for_the_same_track():
    frames = [[(1, box(100), True)]] * 30 + [[(1, box(100), False)]] * 50 + [[(1, box(100), True)]] * 30
    assert len(feed(EventEngine(CFG["events"], "cam", FPS), frames)) == 1


def zone_set(tmp_path):
    f = tmp_path / "z.json"
    f.write_text('{"camera_id": "cam", "frame_size": [1920, 1080], '
                 '"zones": [{"zone_id": "z", "polygon": [[500, 500], [900, 500], [900, 800], [500, 800]]}]}')
    return ZoneSet(f, (1920, 1080), CFG["zones"], FPS)


def test_zone_uses_feet_not_box_overlap(tmp_path):
    zs = zone_set(tmp_path)
    # box overlaps the zone but the feet are 40 px below its bottom edge: outside
    assert zs.read(1, [600, 640, 660, 840], 0.9)[0]["z"][0] is False
    # feet well inside
    assert zs.read(2, [600, 500, 660, 700], 0.9)[0]["z"][0] is True


def test_zone_edge_hysteresis(tmp_path):
    zs = zone_set(tmp_path)
    feet = lambda tid, y: zs.read(tid, [600, y - 180, 660, y], 0.9)[0]["z"][0]  # noqa: E731
    # entering and leaving both need 0.1 x 180 = 18 px past the 800 px edge, so feet swaying between
    # 790 and 805 keep whatever state the person already had
    assert [feet(1, y) for y in (760, 790, 805, 790, 805)] == [True] * 5
    assert [feet(2, y) for y in (840, 805, 790, 805, 790)] == [False] * 5


def test_unconfirmed_person_alerts_only_once_confirmed(tmp_path):
    zs = zone_set(tmp_path)
    zs.read(1, [600, 500, 660, 700], 0.35)
    assert not zs.confirmed(1)  # only ever a weak detection, like a pattern in a grate
    zs.read(1, [600, 500, 660, 700], 0.8)
    assert zs.confirmed(1)
    # inside for 2 s while unconfirmed: no alert, but the evidence counts, so the first confirmed
    # frame alerts at once instead of starting the 0.5 s persistence from scratch
    key = ("zone_entry", "z")
    engine = EventEngine(CFG["events"], "cam", FPS)
    held = [engine.step(i, i / FPS, [Observation(1, box(100), {key: (True, 0.4)}, frozenset([key]))])
            for i in range(20)]
    assert held == [[]] * 20
    assert len(engine.step(20, 2.0, [Observation(1, box(100), {key: (True, 0.8)})])) == 1


def test_hidden_feet_use_remembered_height(tmp_path):
    zs = zone_set(tmp_path)
    zs.read(1, [600, 400, 660, 700], 0.9)                      # full body seen once: 300 px tall
    reading, (pt, hidden) = zs.read(1, [600, 420, 660, 560], 0.9)  # cart hides the legs
    assert hidden and abs(pt[1] - (420 + 300 * CFG["zones"]["height_decay_per_s"] ** (1 / FPS))) < 2
    assert reading["z"][0] is True
