# Questions for the Jetson AGX / streaming engineer

Context for why we ask: we (CV) have a 4-camera cross-court player-tracking pipeline
(detector → ByteTrack → jersey OCR → pose → cross-camera identity correction) that
today runs offline on recorded footage. We want to run it live inside the new
DeepStream/NDI setup on the Orin AGX (`bauer-san/uspaces-docker` →
`bauersan/jetson-ndi-yolo`). The old rig was GoPro HERO12s on USB to two Orin Nanos
(`gopro-automation-linux`), record → S3 → cloud jobs. These questions close the gap
between what the Dockerfile implies and what we need to commit to a design.

## A. Cameras & streams
1. Which cameras replaced the GoPros, and which capture path is live in production:
   **NDI (`ndisrc`, the Zowietek tools?) or GigE Vision (`aravissrc`/AJA)?** The image
   builds both.
2. Per stream: resolution, frame rate, codec/format (full NDI vs NDI|HX?), and
   bandwidth. Are all 4 court angles on one switch/VLAN to a single AGX?
3. **Sync**: are the 4 streams genlocked or NTP/PTP-timestamped? Do NDI timestamps
   give us reliable cross-camera alignment (today we audio-sync to ±1-2 frames —
   do the new streams even carry audio)?
4. Where is the camera-ID → court-angle (FL/FR/NL/NR) mapping configured now
   (old rig used `CAMERA_ANGLE_MAP` in `.env`)?
5. Are mounts/optics fixed at the same 4 positions as the calibrated GoPro rig?
   Any PTZ/auto-exposure/auto-focus features enabled that could shift framing
   mid-game? (Our court homographies assume a fixed view; new cameras ⇒ we need a
   calibration session anyway — when can we get one?)

## B. Start/stop & game lifecycle
6. What triggers streaming/recording start and stop now? (Old flow: Flask REST on
   the Jetson driven by the booking/frontend + `game_auto_end.py`.) Is there still
   an API we can subscribe to, or is it manual via `zowiecam_tui.py`?
7. Is footage still **recorded** while streaming (local disk? S3 chapters like
   before?), or is NDI the only sink? We need the recording for the accuracy
   (finalization) pass and for training-data flywheel.
8. At stream-start time, do we know **game_id and roster** (Supabase `games` row
   created before tip-off)? Our identity layer needs the roster (numbers + dual
   numbers per kit) at pipeline start.
9. What marks game END, and what's the expected turnaround for post-game outputs?

## C. Compute & deployment
10. AGX config: JetPack/L4T version, power mode (MAXN?), RAM, storage. Is the
    DeepStream container the only workload on the box, or does it share with
    recording/upload services?
11. Is it **one AGX for all 4 streams** per court, or split across devices like the
    old dual-Nano setup? Any measured GPU/decoder headroom with 4 streams live?
12. How do we ship code/models: the GitHub-Actions → Docker Hub flow rebuilds on
    commit — who owns merges, and can we add our models (TensorRT engines,
    OCR/pose weights) to that image or mount them as a volume?
13. Is a YOLO already running live in DeepStream today (the image compiles the
    DeepStream-Yolo parser)? Which model, what task, and its measured fps — so we
    know the baseline the box sustains.

## D. Where our pipeline plugs in
14. What's the intended CV hook: `pyds` pad-probe callbacks on the metadata, a
    reference `deepstream-app` config we should extend, or raw frames via appsink?
    Is there an example pipeline config in `uspaces-tools`?
15. Any constraints already hit with multi-model pipelines (primary detector +
    secondary nets for pose/OCR crops): nvinfer batch sizes, memory, plugin
    versions?
16. Output side: where should live results land — Firebase `basketball-games.logs`
    like shot detection, Supabase plays, a WebSocket to the frontend? What latency
    does the product actually need (2s? 10s? end of play?)?
17. Can we persist per-game artifacts on the box (detections, OCR anchors) and
    upload post-game for the finalization pass — disk budget and S3 path
    conventions?

## E. Operations
18. Stream-loss behavior: NDI reconnect handling, and what should inference do when
    a camera drops (we can run 3-camera degraded, but need the event signal)?
19. Monitoring: is `ndi_stats.py`/anything exporting stream health? Alerting path
    (CloudWatch like the CV runbook, or something new)? Remote access still
    Tailscale?
20. Clock discipline on cameras + AGX (NTP/PTP)? Cross-camera identity depends on
    frame-level time alignment.
21. Scale plan: how many courts/venues will one image serve, and does the design
    assume one AGX per court?

## What we'll do with the answers
- Confirm the two-pass design: LIVE pass on AGX (TensorRT detector + tracker +
  triggered jersey-OCR → provisional identities in seconds) + FINALIZATION pass
  (full cross-camera anchor interpolation → accuracy-grade output) on the recording
  or a rolling delay.
- Size the AGX budget: detector engine fps × 4 streams + OCR/pose secondaries.
- Schedule camera calibration for the new rig.
