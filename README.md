# FoveaCast (local, INT8): where do people look on this UI?

Models: FoveaCast v0.2.0 INT8 (MSI-Net fine-tuned on UEyes), one per viewing window: 1s first glance,
3s quick scan, 7s full viewing. Source: https://github.com/khawkins98/foveacast-training/releases/tag/v0.2.0
(downloaded 2026-10-19, ~32 MB each). INT8 vs FP32: CC/NSS within 0.1%, KLD 1.7-2.6% worse.

    uv run -q --project ~/foveacast python ~/foveacast/attention.py shot.png [more.png] \
        [--durations 1s,3s,7s] [--regions regions.json] [--out DIR]

regions.json: `[{"name": "Primary CTA", "box": [x, y, w, h]}]` in the image's own pixels.
Output per image: heatmap overlays per duration, a side-by-side strip, and JSON with the top hotspots
and each region's attention share, rank and `vs_area` (share divided by area share; >1 = draws more than its size).
CPU only, ~3 s for three durations on a 2880x1800 image. Used by the `interaction-agent` subagent.

Limits: predicts visual attention, not clicks; model sees 320x240 (tiny text/icons unresolved); trained on
mostly Western, light-mode 2020-22 UIs (dark UIs and non-Latin scripts are less reliable); one viewport at a time.

Attribution (required if results are published): UEyes, Jiang et al., CHI '23, doi:10.1145/3544548.3581096
(CC BY 4.0); MSI-Net, Kroner et al., Neural Networks 2020 (MIT); FoveaCast by Ken Hawkins (MIT).

Setup: the `.onnx` files are not in this repo. Download `foveacast-v3-{1s,3s,7s}-int8.onnx` from the release
page above into `models/`, then run the command above (needs [uv](https://docs.astral.sh/uv/)).
The `.parity.json` / `.quality.json` files are the upstream metrics for each model.
