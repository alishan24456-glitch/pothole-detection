# Pothole Detection — ADAS / road-maintenance road-surface monitor

From a forward-facing (dashcam) view, dark irregular road-surface
blobs inside the ego lane are found and tracked across frames,
graded **MINOR / MAJOR** by apparent size, with a live pothole
count, a per-pothole severity board and a timestamped event log.

## How it works

Classical OpenCV — no learned model, no YOLO weights needed.

1. **Road ROI.** A trapezoid covering the road ahead (the lower
   half of the frame, converging toward the vanishing point). The
   trapezoid also keeps the detector off the pavement, parked
   tyres and shop fronts at the frame edges.
2. **Dark-blob detection.** Each frame is compared against a
   heavily blurred copy of itself: a patch clearly darker than
   its local surroundings is a candidate. The background estimate
   combines a wide Gaussian blur with a large-kernel
   morphological closing — the closing matters, because a big dark
   pothole darkens its own Gaussian background estimate, so only
   its rim would exceed the threshold.
3. **Shape filters.** A pothole seen in perspective is a flat,
   wider-than-tall patch on the ground, so a blob must be wider
   than it is tall (wheels, legs and posts are ~1:1 or taller and
   are rejected on shape alone). Area, solidity and extent are
   also filtered, and a blob wider than 45% of the road at its
   height is a shadow band, not a pothole.
4. **Tracking + motion gate.** Centroid matching across frames. A
   candidate confirms only after 8 consecutive sightings (~0.27 s)
   AND a sideways-motion gate: its centroid velocity must not be
   mostly sideways (a passing motorcycle slides across the frame;
   a pothole rushes down toward the camera or sits still). A
   candidate that fails the gate is rejected permanently and
   logged once.
5. **Severity.** By apparent size — the largest area a pothole
   reaches while tracked. Below 3,500 px² it is MINOR, at or above
   it MAJOR; an upgrade MINOR → MAJOR is logged when it happens.

## Demo — what the real footage actually shows

Demo: `files/project-demos/pothole-detection-demo.mp4` (30 s,
1280×720) plus the H.264 LinkedIn copy in
`files/project-demos/linkedin-h264/pothole-detection-demo.mp4`.

Footage: Pexels, "Africa Cameroon Douala Road" (video 34218229),
a street-level forward view down a badly damaged road in Douala,
Cameroon — muddy, water-filled potholes across the road surface,
with motorcycles passing in front of the camera. Downloaded fresh
for this build under the 14 Nov fresh-video rule; never used by
any earlier project. The clip is 19.7 s, resampled to 30 fps, and
the demo loops it to 30 s (loop point 19.7 s, logged in the event
log; tracks reset there).

Measured on this clip (30 s, looped, final build):

- 901 frames, 1,150 dark-blob candidates (1.28 per frame)
- **44 potholes confirmed: 38 MINOR, 6 MAJOR**; 2 severity
  upgrades MINOR → MAJOR; 13 sideways candidates rejected
  (passing motorcycles, at 62–285 px/s lateral)
- Largest area seen: 28,104 px² (4.19% of the frame)
- 4 live potholes at the end; 99 events logged

Read the numbers honestly:

- **The count is track instances, not 44 distinct potholes.**
  The camera is nearly static and the traffic is congested, so a
  passing motorcycle hides a patch for a second and the tracker
  re-counts it under a new ID when it reappears. One physical
  patch can appear two or three times in the count.
- **The big MAJORs are merged damaged areas, not single holes.**
  This road is one long stretch of connected muddy damage, so
  adjacent patches merge into one large blob (the largest,
  28,104 px², is a whole damaged stretch). On this clip MAJOR
  means "large damaged area" more than "one big hole".
- **Shadows read the same as holes.** The biggest MAJOR on the
  first pass is partly the dark shadow under a parked motorcycle
  (visible around 8 s in the demo), not a hole at all. A dark
  patch is a candidate, not a confirmed pothole.
- The camera is nearly static in congested traffic, not a moving
  dashcam — so potholes sit still instead of rushing toward the
  camera. The downward-rush behaviour is verified in simulation
  only (see below).
- Water-filled holes that reflect the sky are brighter than the
  road and are missed by design; this detector only sees holes
  darker than the asphalt around them. Severity is apparent size
  in the image, never measured depth.

## Simulation — what was verified there, not on real footage

`--simulate` renders a driver's-seat road with three potholes of
known size driving toward the camera, plus a dark bar sliding
sideways across the road that must never confirm. The detector
reads only the rendered pixels.

- Small hole → confirmed MINOR (max area 2,932 px², under the
  3,500 threshold); medium hole → confirmed MINOR then upgraded
  to MAJOR (max 8,795 px²); large hole → confirmed MINOR then
  upgraded to MAJOR (max 26,722 px²)
- The sideways bar was rejected twice (105 and 193 px/s lateral,
  0 px vertical) and never confirmed
- 901 frames, 827 candidates, 3 confirmed (1 MINOR, 2 MAJOR),
  2 upgrades, 2 sideways rejections

The downward rush toward the camera, the MINOR → MAJOR upgrade
on approach, and the sideways rejection are all verified in
simulation; on the real clip the camera is static, so only the
stationary-patch case is exercised there.

## Bugs found during verification

1. **Headless crash:** `cv2.destroyAllWindows()` raises
   `cv2.error` under `opencv-python-headless` (no GUI backend),
   which aborted the run *after* the video was written but
   *before* the event log and summary were written. Fixed by
   wrapping it in `try/except cv2.error`, as the
   lane-departure-warning project does.
2. **Background darkening:** with a Gaussian-blur background
   alone, a large dark pothole darkened its own background
   estimate, so only its rim exceeded the threshold — the medium
   simulated hole read 2,401 px² instead of ~10,000 and its rim
   split into phantom candidates. Fixed by taking the brighter of
   a Gaussian background and a large-kernel morphological-closing
   background.
3. **Range gate too weak:** the first sideways gate judged the
   centroid range over the 8-frame confirmation window, which is
   small even for a fast slider — the decoy bar (195 px/s)
   confirmed as a MAJOR pothole. Fixed by judging centroid
   *velocity* instead.
4. **Rejected-then-confirmed:** the first build only logged a
   gate failure and kept judging the track, so a motorcycle that
   slowed down in traffic later confirmed as a pothole — the same
   ID appeared as both "rejected" and "confirmed" in the event
   log. Fixed: a gate failure now rejects the track permanently.
5. **Short tracks re-counted:** a passing motorcycle hiding a
   static patch for ~1 s killed the track, so the same patch was
   re-counted. Fixed by holding tracks for 30 missed frames (1 s)
   instead of 12.
6. **Event text ran off the frame:** the latest-event line in the
   bottom panel was cut off at the right edge. Fixed by
   truncating it to fit.

## Usage

```bash
pip install -r requirements.txt
python pothole_detection.py                  # opens a file picker
python pothole_detection.py --source pick    # same picker, explicit
python pothole_detection.py --source road.mp4
python pothole_detection.py --source 0       # webcam
python pothole_detection.py --simulate --max-seconds 30 --no-display
```

Press Q to quit. The video is written to `output.mp4` (or
`--output`), the timestamped event log to
`output_events.log` and the final counters to
`output_summary.txt` next to it. `--loop` loops a short source
until `--max-seconds` is reached (tracks reset at the loop point,
logged).

## Honest limits

- A dark patch is a **candidate**, not a confirmed pothole.
  Shadows, patches, drains, wet mud and the dark gap between
  pavement slabs read the same.
- Severity is **apparent size**, not measured depth. A small hole
  close to the camera reads bigger than a big hole far away.
- Dry, light-coloured potholes and sky-reflecting water-filled
  holes are missed by design — only patches darker than their
  surroundings are found.
- On congested, nearly static footage the confirmed count
  over-counts: one physical patch can be counted more than once
  when a passing vehicle hides it.
- The detector reads the road surface only. It does not read lane
  lines (see lane-departure-warning) or vehicles (see
  tailgating-detection / forward-collision-warning).

Author: Abdul Hanan — Computer Vision & AI Developer
