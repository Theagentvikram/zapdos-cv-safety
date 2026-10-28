"""Annotated frame rendering. Only person, helmet and safety-vest are drawn."""

from __future__ import annotations

import cv2

# BGR
GREEN = (0, 190, 0)
YELLOW = (0, 215, 255)
RED = (0, 0, 235)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def state_label(state: str, absence_evidence: bool) -> tuple[str, tuple]:
    if state.startswith("SUSPECTED"):
        return "SUSPECTED OFF", RED
    if state.endswith("_ON_HELD"):
        return "ON (held)", YELLOW
    if state.endswith("_ON"):
        return "ON", GREEN
    return ("UNKNOWN*" if absence_evidence else "UNKNOWN"), YELLOW


def _text_box(img, x, y, text, bg, fg=BLACK, scale=0.55):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 6, y + 2), bg, -1)
    cv2.putText(img, text, (x + 3, y - base), FONT, scale, fg, 1, cv2.LINE_AA)
    return x + tw + 6


def draw_worker(img, track_id, box, helmet, vest, helmet_box=None, vest_box=None):
    """helmet / vest: (state, absence_evidence)."""
    h_text, h_color = state_label(*helmet)
    v_text, v_color = state_label(*vest)
    if RED in (h_color, v_color):
        box_color = RED
    elif h_color == GREEN and v_color == GREEN:
        box_color = GREEN
    else:
        box_color = YELLOW
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    for ppe_box, color, tag in ((helmet_box, h_color, "H"), (vest_box, v_color, "V")):
        if ppe_box is None:
            continue
        a1, b1, a2, b2 = (int(v) for v in ppe_box)
        cv2.rectangle(img, (a1, b1), (a2, b2), color, 2)
        _text_box(img, a1, b1 - 1, f"{tag}{track_id:02d}", color, scale=0.4)
    ty = max(20, y1 - 4)
    x = _text_box(img, x1, ty, f"ID {track_id:02d}", WHITE)
    x = _text_box(img, x, ty, f"Helmet: {h_text}", h_color)
    _text_box(img, x, ty, f"Vest: {v_text}", v_color)


def draw_candidate(img, label, box):
    """Unconfirmed track or untracked person detection: thin gray box, no PPE state."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), GRAY, 1)
    _text_box(img, x1, max(14, y1 - 2), label, GRAY, scale=0.4)


def draw_header(img, camera_id, t, helmet_validated, vest_validated):
    lines = [
        f"{camera_id}  t={t:6.2f}s  SH17 YOLOv8s + ByteTrack",
        f"absence alerts: helmet {'VALIDATED' if helmet_validated else 'not validated'}, "
        f"vest {'VALIDATED' if vest_validated else 'not validated'}",
        "UNKNOWN* = absence evidence met but camera not validated (no alert)",
    ]
    for i, line in enumerate(lines):
        _text_box(img, 8, 26 + i * 24, line, BLACK, WHITE, scale=0.55)
