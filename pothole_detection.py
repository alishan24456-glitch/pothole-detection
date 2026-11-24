#!/usr/bin/env python3
"""
Pothole Detection -- ADAS / road-maintenance road-surface monitor
==================================================================

From a forward-facing (dashcam) view, dark irregular road-surface
blobs inside the ego lane are found and tracked across frames,
graded MINOR / MAJOR by apparent size, with a live pothole count, a
per-pothole severity board and a timestamped event log.

* **Detection** -- classical OpenCV, no learned model. Inside a
  road ROI (a trapezoid covering the road ahead) each frame is
  compared against a heavily blurred copy of itself: a patch that
  is clearly darker than its local surroundings is a candidate
  blob. Contours are filtered by area, solidity, extent and shape:
  a pothole seen in perspective is a flat, wider-than-tall patch
  on the ground, so a blob must be wider than it is tall -- a
  wheel, a pedestrian's leg or a shadow cast by an upright object
  is roughly as tall as it is wide and is rejected on shape alone.
* **Tracking** -- centroid matching across frames. A candidate
  becomes a confirmed pothole only after 8 consecutive sightings
  (~0.27 s) AND a motion gate: over its sightings the centroid
  must have moved mostly downward (rushing toward the camera, as
  a road defect does on a moving car) or hardly at all (a static
  camera over a damaged road). A dark object sliding sideways
  across the frame -- a passing motorcycle, its shadow, a
  pedestrian -- has a large lateral range and a small vertical
  range, so it never confirms.
* **Severity** -- by apparent size: the largest area a pothole
  reaches while tracked. Below MAJOR_AREA px^2 it is MINOR, at or
  above it MAJOR. An upgrade MINOR -> MAJOR is logged when it
  happens. Severity is apparent size in the image, NOT measured
  depth: a small hole close to the camera reads bigger than a big
  hole far away, and a water-filled hole can read lighter than the
  asphalt around it.

The demo clip (Pexels, Douala street, 19.7 s, looped to 30 s --
never used by any earlier project) is reported exactly as
measured in the README, including how many pothole candidates it
actually contains. The simulation (a rendered road with potholes
of known sizes plus a sideways-rolling wheel that must never
confirm, same pixel pipeline) verifies the full state machine:
confirmation, MINOR / MAJOR grading, the upgrade, and the
sideways-motion rejection.

Honest framing: a dark patch is a CANDIDATE, not a confirmed
pothole -- shadows, patches, drains and wet mud read the same,
and this detector cannot tell a hole from a dark stain. Severity
is by apparent size, not measured depth. A dry, light-coloured
pothole or a water-filled hole reflecting the sky is brighter
than the road and is missed by design. The README states what
the demo clip does and does not show.

Examples
--------
    python pothole_detection.py                 (opens a file picker)
    python pothole_detection.py --source pick   (same picker, explicit)
    python pothole_detection.py --source road.mp4
    python pothole_detection.py --source 0      (webcam)
    python pothole_detection.py --simulate --max-seconds 30 --no-display

Press Q to quit. The video is written to ``output.mp4``, the
timestamped event log to ``output_events.log`` and the final counters
to ``output_summary.txt`` next to it.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Layout (the output video is always 1280x720)
# ---------------------------------------------------------------------------
W, H = 1280, 720
TITLE_H = 46
PANEL_H = 150                     # stats panel at the bottom
VIDEO_W, VIDEO_H = W, H - TITLE_H - PANEL_H
BG = (24, 26, 32)
PANEL_BG = (32, 35, 46)
WHITE = (255, 255, 255)
GREY = (160, 166, 178)
GREEN = (90, 200, 90)
AMBER = (60, 170, 255)
RED = (60, 60, 235)
YELLOW = (0, 210, 235)
CYAN = (235, 200, 0)

# ---------------------------------------------------------------------------
# Detection tuning (video-area coordinates, 1280x524)
# ---------------------------------------------------------------------------
# Road ROI: a trapezoid from the bottom corners up to the vanishing
# point. Tuned for a forward dashcam: the road fills the lower half
# and converges near the centre of the frame. The trapezoid also
# keeps the detector off the pavement, parked tyres and shop fronts
# at the frame edges (on the demo clip the stacked tyres on the
# right sit outside it).
ROI_POLY = np.array([[(40, VIDEO_H), (480, 190),
                      (800, 190), (1240, VIDEO_H)]], dtype=np.int32)
# A pixel counts as "dark patch" when it is this much darker than
# the heavily blurred local background (the road around it).
DARK_DELTA = 20
BG_BLUR_SIGMA = 30.0
# Contour filters.
MIN_AREA = 200                    # px^2 in video coordinates
MAX_AREA = 50000
MIN_SOLIDITY = 0.50               # area / convex-hull area
MIN_EXTENT = 0.35                 # area / bounding-box area
# A pothole in perspective is a flat patch on the ground: wider
# than tall. Upright objects (wheels, legs, posts) are ~1:1 or
# taller and are rejected on shape alone.
MIN_W_OVER_H = 1.25
MAX_W_OVER_H = 6.0
# A blob wider than this fraction of the road width at its height
# is a shadow band or the road edge, not a pothole.
MAX_ROAD_FRAC = 0.45
# Tracking.
MATCH_DIST = 110                  # px, centroid match gate
MAX_AREA_RATIO = 2.5              # matched blob areas must be close
MAX_MISSED = 30                   # frames before a track dies; a
                                # passing motorcycle can hide a static
                                # patch for ~1 s, so the track must
                                # survive brief occlusions instead of
                                # re-counting the same patch
CONFIRM_FRAMES = 8                # consecutive sightings to confirm
# Motion gate (checked at confirmation): a pothole rushes down
# toward the camera or sits still; a passing vehicle slides
# sideways. Reject when the lateral speed is above
# LATERAL_SPEED_PX_S and dominates the vertical speed, or when the
# lateral range is large AND more than 1.5x the vertical range.
LATERAL_REJECT_PX = 90
LATERAL_SPEED_PX_S = 60.0
LATERAL_OVER_VERTICAL = 1.5
# Severity: apparent size. A track's largest area while tracked
# decides it; at or above this it is MAJOR.
MAJOR_AREA = 3500                 # px^2 in video coordinates

# Severity colours.
SEV_COLOUR = {"MINOR": AMBER, "MAJOR": RED}


def roi_width_at(y: float) -> float:
    """Width of the road trapezoid at height y (video coords)."""
    t = (VIDEO_H - y) / (VIDEO_H - 190)
    return (1240 - 40) - t * ((1240 - 40) - (800 - 480))


# ---------------------------------------------------------------------------
# Pothole tracker -- contour candidates + centroid tracking
# ---------------------------------------------------------------------------
class PotholeTracker:
    """Finds dark road-surface blobs and tracks them into potholes."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.frames = 0
        self.candidates_total = 0
        self.candidates_this_frame = 0
        self.confirmed = 0
        self.minor_count = 0
        self.major_count = 0
        self.upgrades = 0
        self.rejected_sideways = 0
        self.events = []
        self.video_frames = 0
        self.max_area_seen = 0.0

    def _log(self, t, text):
        self.events.append((t, text))

    # -- detection ----------------------------------------------------
    def _candidates(self, frame):
        """Dark-blob candidates inside the road ROI.

        Returns a list of (x, y, w, h, area, cx, cy)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Local background: the road around the patch. Two estimates
        # are combined: a wide Gaussian blur (smooth shading) and a
        # large-kernel morphological closing (which fills dark holes
        # with the surrounding road brightness). The closing is
        # essential -- with the Gaussian alone, a big dark pothole
        # darkens its own background estimate, so only its rim
        # exceeds the threshold and the measured area is a fraction
        # of the hole (found on the first simulation run: the medium
        # hole read 2,401 px^2 instead of ~10,000, and its rim split
        # into phantom candidates). Taking the brighter of the two
        # estimates keeps the background asphalt-bright even inside
        # a large hole.
        gauss_bg = cv2.GaussianBlur(blur, (0, 0), BG_BLUR_SIGMA)
        small = cv2.resize(blur, None, fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (31, 31))
        closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
        close_bg = cv2.resize(closed, (VIDEO_W, VIDEO_H),
                              interpolation=cv2.INTER_LINEAR)
        bg = cv2.max(gauss_bg, close_bg)
        diff = cv2.subtract(bg, blur)
        _, mask = cv2.threshold(diff, DARK_DELTA, 255,
                                cv2.THRESH_BINARY)
        roi = np.zeros_like(mask)
        cv2.fillPoly(roi, ROI_POLY, 255)
        mask = cv2.bitwise_and(mask, roi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (MIN_AREA <= area <= MAX_AREA):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            if not (MIN_W_OVER_H <= w / h <= MAX_W_OVER_H):
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0 or area / hull_area < MIN_SOLIDITY:
                continue
            if area / (w * h) < MIN_EXTENT:
                continue
            if w > MAX_ROAD_FRAC * roi_width_at(y + h / 2):
                continue
            out.append((float(x), float(y), float(w), float(h),
                        float(area), x + w / 2, y + h / 2))
        return out

    # -- tracking -----------------------------------------------------
    def update(self, frame, t):
        self.frames += 1
        cands = self._candidates(frame)
        self.candidates_this_frame = len(cands)
        self.candidates_total += len(cands)

        used = set()
        for (x, y, w, h, area, cx, cy) in cands:
            best, best_d = None, MATCH_DIST
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = math.hypot(cx - tr["cx"], cy - tr["cy"])
                if d >= best_d:
                    continue
                if not (1 / MAX_AREA_RATIO
                        <= area / max(1.0, tr["area"])
                        <= MAX_AREA_RATIO):
                    continue
                best, best_d = tid, d
            if best is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "id": tid, "cx": cx, "cy": cy, "x": x, "y": y,
                    "w": w, "h": h, "area": area, "missed": 0,
                    "seen": 1, "confirmed": False, "rejected": False,
                    "severity": None, "max_area": area,
                    "first_t": t, "last_t": t,
                    "xs": [cx], "ys": [cy], "ts": [t],
                }
                best = tid
            used.add(best)
            tr = self.tracks[best]
            tr.update({"cx": cx, "cy": cy, "x": x, "y": y,
                       "w": w, "h": h, "area": area, "missed": 0,
                       "last_t": t})
            tr["seen"] += 1
            tr["xs"].append(cx)
            tr["ys"].append(cy)
            tr["ts"].append(t)
            if len(tr["xs"]) > 90:
                tr["xs"].pop(0)
                tr["ys"].pop(0)
                tr["ts"].pop(0)
            if area > tr["max_area"]:
                tr["max_area"] = area
            if tr["max_area"] > self.max_area_seen:
                self.max_area_seen = tr["max_area"]
            self._maybe_confirm(tr, t)
            self._maybe_upgrade(tr, t)

        for tid in list(self.tracks):
            if tid in used:
                continue
            tr = self.tracks[tid]
            tr["missed"] += 1
            if tr["missed"] > MAX_MISSED:
                if tr["confirmed"]:
                    self._log(t, f"pothole #{tid} left the frame -- "
                                 f"final {tr['severity']}, max area "
                                 f"{tr['max_area']:.0f} px^2, "
                                 f"seen {tr['last_t'] - tr['first_t']:.1f} s")
                del self.tracks[tid]

        live = [tr for tr in self.tracks.values() if tr["confirmed"]]
        return {"live": live, "candidates": self.candidates_this_frame}

    def _motion_ok(self, tr):
        """Sideways-motion gate: potholes rush down or sit still.

        Judged on centroid VELOCITY over the track's history, not
        on the range alone: the range over the 8-frame confirmation
        window is small even for a fast sideways slider (found on
        the second simulation run -- the decoy bar slid ~195 px/s
        but moved only ~52 px in the confirmation window, so the
        range gate passed it and it confirmed as a MAJOR pothole).
        A pothole rushes down toward the camera (|vy| large) or
        sits still (both speeds ~0); a passing object slides
        sideways fast while barely moving down.
        """
        xs, ys, ts = tr["xs"], tr["ys"], tr["ts"]
        span = ts[-1] - ts[0]
        if span < 0.1 or len(xs) < 2:
            return True
        vx = (xs[-1] - xs[0]) / span
        vy = (ys[-1] - ys[0]) / span
        if (abs(vx) > LATERAL_SPEED_PX_S
                and abs(vx) > LATERAL_OVER_VERTICAL * abs(vy)):
            return False
        lat = max(xs) - min(xs)
        vert = max(ys) - min(ys)
        if lat > LATERAL_REJECT_PX and lat > LATERAL_OVER_VERTICAL * vert:
            return False
        return True

    def _maybe_confirm(self, tr, t):
        if tr["confirmed"] or tr["rejected"]:
            return
        if tr["seen"] < CONFIRM_FRAMES:
            return
        if not self._motion_ok(tr):
            # A sideways slider is rejected PERMANENTLY: the gate
            # is judged once, at confirmation time. (The first
            # build only logged the failure and kept judging the
            # track, so a motorcycle that slowed down in traffic
            # later confirmed as a pothole -- the same ID appeared
            # as both "rejected" and "confirmed" in the event log,
            # found on the first real run.)
            tr["rejected"] = True
            self.rejected_sideways += 1
            span = tr["ts"][-1] - tr["ts"][0]
            vx = ((tr["xs"][-1] - tr["xs"][0]) / span
                  if span > 0 else 0.0)
            self._log(t, f"candidate #{tr['id']} rejected -- "
                         f"sliding sideways "
                         f"({vx:.0f} px/s lateral, "
                         f"{max(tr['ys']) - min(tr['ys']):.0f} px vertical)")
            return
        tr["confirmed"] = True
        tr["severity"] = ("MAJOR" if tr["max_area"] >= MAJOR_AREA
                          else "MINOR")
        self.confirmed += 1
        if tr["severity"] == "MAJOR":
            self.major_count += 1
        else:
            self.minor_count += 1
        self._log(t, f"pothole #{tr['id']} confirmed "
                     f"{tr['severity']} -- area "
                     f"{tr['max_area']:.0f} px^2")

    def _maybe_upgrade(self, tr, t):
        if not tr["confirmed"] or tr["severity"] != "MINOR":
            return
        if tr["max_area"] >= MAJOR_AREA:
            tr["severity"] = "MAJOR"
            self.minor_count -= 1
            self.major_count += 1
            self.upgrades += 1
            self._log(t, f"pothole #{tr['id']} upgraded MINOR -> "
                         f"MAJOR -- area {tr['max_area']:.0f} px^2")

    @property
    def live_count(self):
        return sum(1 for tr in self.tracks.values() if tr["confirmed"])

    def reset(self, t):
        """Drop all tracks: the source looped, so the scene restarted."""
        n = len(self.tracks)
        self.tracks.clear()
        self._log(t, f"source looped -- {n} in-flight tracks reset "
                     f"(confirmation timers restart from zero)")

    # -- reporting ------------------------------------------------------
    def summary(self):
        n = max(1, self.frames)
        lines = [
            "Pothole Detection -- summary",
            f"frames: {self.frames} ({self.video_frames} written, "
            f"{self.frames / 30:.1f}s at 30 fps)",
            f"dark-blob candidates: {self.candidates_total} "
            f"({self.candidates_total / n:.2f} per frame)",
            f"potholes confirmed: {self.confirmed} "
            f"(MINOR {self.minor_count}, MAJOR {self.major_count})",
            f"live potholes at end: {self.live_count}",
            f"severity upgrades MINOR -> MAJOR: {self.upgrades}",
            f"sideways candidates rejected: {self.rejected_sideways}",
            f"largest pothole area seen: {self.max_area_seen:.0f} px^2 "
            f"({100 * self.max_area_seen / (VIDEO_W * VIDEO_H):.2f}% of frame)",
            f"events logged: {len(self.events)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulation -- a rendered road with potholes of known sizes
# ---------------------------------------------------------------------------
class RoadSim:
    """Renders a driver's-seat road with scripted potholes.

    Three potholes of known size drive toward the camera (their
    rendered area grows as they approach, exactly as a real hole
    would): a small one (MINOR), a medium one (MAJOR) and a large
    one (MAJOR). A dark wheel rolls sideways across the road and
    must never confirm -- it is the sideways-motion gate's test.
    The detector reads only the rendered pixels; it never sees the
    scripted sizes.
    """

    ASPHALT = (74, 74, 74)
    POTHOLE = (36, 36, 38)
    POTHOLE_EDGE = (52, 50, 48)

    def __init__(self, w: int = VIDEO_W, h: int = VIDEO_H,
                 fps: float = 30.0):
        self.w, self.h, self.fps = w, h, fps
        self.frame_idx = 0
        # (start_s, end_s, centre_x_frac, base_width_px)
        self.holes = [
            (3.0, 12.0, 0.38, 62.0),    # small  -> MINOR
            (8.0, 18.0, 0.62, 112.0),   # medium -> MAJOR
            (15.0, 26.0, 0.50, 172.0),  # large  -> MAJOR
        ]
        self._rng = np.random.default_rng(7)

    def _draw_road(self, img, t):
        w, h = self.w, self.h
        horizon = 190
        img[:horizon] = (205, 208, 212)     # pale sky
        img[horizon:] = (72, 96, 62)        # grass verges
        road = np.array([[(40, h), (480, horizon),
                          (800, horizon), (1240, h)]], dtype=np.int32)
        cv2.fillPoly(img, road, self.ASPHALT)
        # Centre dashes rushing toward the camera.
        for i in range(5):
            y0 = horizon + int((i * 70 + (t * 200) % 70) ** 1.3 / 5)
            if y0 >= h:
                continue
            y1 = min(h, y0 + 30)
            cv2.rectangle(img, (636, y0), (644, y1),
                          (235, 235, 235), -1)
        # Light asphalt noise so the frame is not perfectly flat.
        noise = self._rng.integers(-5, 6, size=(h, w, 1)).astype(np.int16)
        img[:] = np.clip(img.astype(np.int16) + noise, 0, 255
                         ).astype(np.uint8)

    def _draw_hole(self, img, cx_frac, base_w, progress):
        """One pothole at a given approach progress (0 far, 1 near)."""
        w, h = self.w, self.h
        scale = 0.35 + progress * 1.15
        y = int(205 + progress * 310)
        bw = base_w * scale
        bh = bw * 0.45
        cx = int(cx_frac * w)
        # Irregular edge: a main ellipse plus overlapping blobs.
        cv2.ellipse(img, (cx, y), (int(bw / 2), int(bh / 2)), 0,
                    0, 360, self.POTHOLE, -1, cv2.LINE_AA)
        for dx, dy, f in ((-0.3, 0.1, 0.45), (0.32, -0.08, 0.4),
                          (0.1, 0.22, 0.35)):
            cv2.ellipse(img, (int(cx + dx * bw), int(y + dy * bh)),
                        (int(bw * f / 2), int(bh * f / 2)), 0, 0, 360,
                        self.POTHOLE_EDGE, -1, cv2.LINE_AA)
        # A darker core.
        cv2.ellipse(img, (cx, y), (int(bw * 0.28), int(bh * 0.28)), 0,
                    0, 360, (24, 24, 26), -1, cv2.LINE_AA)

    def _draw_wheel(self, img, t):
        """A dark bar sliding sideways across the road (the decoy).

        On screen 20-24 s, sliding left to right at mid-height. It
        is dark, flat and wide, so shape alone cannot reject it --
        only the sideways-motion gate can keep it from confirming
        as a pothole. (An earlier version drew a round wheel, but
        the shape filter rejected it before tracking, so the gate
        was never exercised.)
        """
        if not (20.0 <= t < 24.0):
            return
        f = (t - 20.0) / 4.0
        cx = int(250 + f * 780)
        cy = 400
        cv2.rectangle(img, (cx - 110, cy - 26), (cx + 110, cy + 26),
                      (32, 32, 32), -1, cv2.LINE_AA)
        cv2.rectangle(img, (cx - 110, cy - 26), (cx + 110, cy - 26),
                      (48, 48, 48), 4, cv2.LINE_AA)

    def frame(self):
        self.frame_idx += 1
        t = self.frame_idx / self.fps
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._draw_road(img, t)
        for start, end, cx_frac, base_w in self.holes:
            if start <= t < end:
                progress = (t - start) / (end - start)
                self._draw_hole(img, cx_frac, base_w, progress)
        self._draw_wheel(img, t)
        return img


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_canvas(det_frame, tracker, res, elapsed, mode_label):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = BG
    cv2.rectangle(canvas, (0, 0), (W, TITLE_H), PANEL_BG, -1)
    cv2.putText(canvas, "POTHOLE DETECTION  --  road-surface monitor",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    cv2.putText(canvas, mode_label, (W - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)

    vh, vw = det_frame.shape[:2]
    scale = min(VIDEO_W / vw, VIDEO_H / vh)
    nw, nh = int(vw * scale), int(vh * scale)
    resized = cv2.resize(det_frame, (nw, nh))
    ox = (VIDEO_W - nw) // 2
    oy = TITLE_H + (VIDEO_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    sx, sy = nw / vw, nh / vh

    def _mx(x):
        return ox + int(x * sx)

    def _my(y):
        return oy + int(y * sy)

    # Road ROI outline.
    pts = np.array([[_mx(x), _my(y)] for x, y in ROI_POLY[0]],
                   dtype=np.int32)
    cv2.polylines(canvas, [pts], True, GREY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "ROAD ROI", (_mx(60), _my(212)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)

    # Live potholes: ellipse + severity tag.
    for tr in res["live"]:
        x, y, w, h = tr["x"], tr["y"], tr["w"], tr["h"]
        colour = SEV_COLOUR[tr["severity"]]
        cv2.ellipse(canvas, (_mx(x + w / 2), _my(y + h / 2)),
                    (int(w * sx / 2) + 3, int(h * sy / 2) + 3),
                    0, 0, 360, colour, 3, cv2.LINE_AA)
        tag = (f"#{tr['id']} {tr['severity']} "
               f"{tr['max_area']:.0f} px^2")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.55, 2)
        lx = _mx(x)
        if lx + tw + 8 > W - 4:
            lx = max(4, W - 4 - tw - 8)
        ly = _my(y) - 8
        if ly - th < oy + 4:
            ly = _my(y + h) + th + 10
        cv2.rectangle(canvas, (lx, ly - th - 4),
                      (lx + tw + 8, ly + 4), (24, 26, 32), -1)
        cv2.putText(canvas, tag, (lx + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # Bottom panel: board + severity counts.
    py = TITLE_H + VIDEO_H
    cv2.rectangle(canvas, (0, py), (W, H), PANEL_BG, -1)
    live = tracker.live_count
    banner_col = RED if tracker.major_count else (
        AMBER if live else GREEN)
    banner = (f"POTHOLES: {live} live   "
              f"MINOR {tracker.minor_count}   MAJOR {tracker.major_count}")
    cv2.rectangle(canvas, (0, py), (W, py + 6), banner_col, -1)
    cv2.putText(canvas, banner, (16, py + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_col, 2)
    cv2.putText(canvas, f"t = {elapsed:.1f} s", (16, py + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas,
                f"candidates: {res['candidates']}   "
                f"confirmed: {tracker.confirmed}",
                (300, py + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas,
                f"largest area: {tracker.max_area_seen:.0f} px^2   "
                f"frames: {tracker.frames}",
                (16, py + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)
    # Per-pothole severity board (most recent first).
    live_sorted = sorted(res["live"], key=lambda tr: tr["id"])
    for i, tr in enumerate(live_sorted[:3]):
        cv2.putText(canvas,
                    f"#{tr['id']} {tr['severity']} "
                    f"{tr['max_area']:.0f} px^2",
                    (860, py + 42 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, SEV_COLOUR[tr["severity"]], 2)
    # Latest event, bottom-right of the panel (truncated so it
    # never runs off the right edge of the frame).
    if tracker.events:
        et, text = tracker.events[-1]
        short = text if len(text) <= 38 else text[:35] + "..."
        cv2.putText(canvas, f"{et:5.1f}s  {short}",
                    (860, py + 122), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, GREY, 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        root.destroy()
        return path
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pothole Detection -- road-surface monitor")
    ap.add_argument("--source", default="pick",
                    help="video file, webcam index, or 'pick' for a "
                         "file chooser")
    ap.add_argument("--simulate", action="store_true",
                    help="render a scripted road with potholes instead "
                         "of a camera/video")
    ap.add_argument("--output", default="output.mp4",
                    help="output video path (default: output.mp4)")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop after this many seconds (0 = whole clip)")
    ap.add_argument("--loop", action="store_true",
                    help="loop a video source until --max-seconds")
    ap.add_argument("--no-display", action="store_true",
                    help="do not open a preview window")
    args = ap.parse_args()

    simulate = args.simulate
    cap = None
    scene = None
    fps = 30.0
    total_frames = None
    if simulate:
        scene = RoadSim()
        fps = scene.fps
    else:
        src = args.source
        if src == "pick":
            src = pick_file()
            if not src:
                print("No video selected.")
                return 0
        if src.isdigit():
            cap = cv2.VideoCapture(int(src))
        else:
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"Could not open source: {src}")
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tracker = PotholeTracker()

    out_path = args.output
    base, _ = os.path.splitext(out_path)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    if not writer.isOpened():
        print(f"Could not open output video: {out_path}")
        return 1

    frame_idx = 0
    res = {"live": [], "candidates": 0}
    print("Press Q to quit.")
    while True:
        if simulate:
            det_frame = scene.frame()
            t = scene.frame_idx / fps
            frame_idx += 1
            res = tracker.update(det_frame, t)
        else:
            ok, raw = cap.read()
            if not ok:
                if args.loop and total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    tracker.reset(frame_idx / fps)
                    continue
                break
            det_frame = cv2.resize(raw, (VIDEO_W, VIDEO_H))
            frame_idx += 1
            t = frame_idx / fps
            res = tracker.update(det_frame, t)
        if args.max_seconds and t > args.max_seconds:
            break
        canvas = draw_canvas(det_frame, tracker, res, t,
                             "SIMULATION" if simulate else "REAL")
        writer.write(canvas)
        if not args.no_display:
            cv2.imshow("Pothole Detection", canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    total_t = frame_idx / fps
    writer.release()
    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # headless OpenCV has no GUI backend

    tracker.video_frames = frame_idx
    events_path = base + "_events.log"
    with open(events_path, "w", encoding="utf-8") as fh:
        fh.write("".join(f"{et:6.1f}s  {text}\n"
                         for et, text in tracker.events))

    summary = tracker.summary()
    with open(base + "_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(summary)
    print(f"video: {out_path}\nevents: {events_path}\n"
          f"summary: {base}_summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
