"""Predict where people look on a UI screenshot (FoveaCast v0.2.0, INT8, MSI-Net fine-tuned on UEyes).

usage: uv run --project ~/foveacast python ~/foveacast/attention.py IMAGE [IMAGE ...]
         [--durations 1s,3s,7s] [--regions regions.json] [--out DIR]

regions.json: [{"name": "Primary CTA", "box": [x, y, w, h]}, ...] in the image's own pixels.
Writes per image: <stem>-<dur>.png (heatmap overlay), <stem>-strip.png (all durations side by side),
<stem>.json (hotspots + region attention shares), and prints the JSON summary.

Attention is predicted visual attention, not clicks. Model input is 320x240, so small text and icons
are below its resolution. Trained on mostly Western, light-mode 2020-22 UIs (see README).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS = Path(__file__).resolve().parent / "models"
W, H = 320, 240
STOPS = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], np.float32)  # inferno-like


def colormap(m):
    x = np.clip(m, 0, 1) * (len(STOPS) - 1)
    i = np.minimum(x.astype(int), len(STOPS) - 2)
    f = (x - i)[..., None]
    return STOPS[i] * (1 - f) + STOPS[i + 1] * f


def predict(sess, img):
    x = np.asarray(img.resize((W, H), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    return np.clip(out, 0, None)


def hotspots(m, n=5, frac=0.5):
    """Connected blobs above frac*peak on the 320x240 map, biggest attention mass first."""
    mask = m >= frac * m.max()
    seen = np.zeros_like(mask)
    blobs = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, pts = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v, u = y + dy, x + dx
                if 0 <= v < H and 0 <= u < W and mask[v, u] and not seen[v, u]:
                    seen[v, u] = True
                    stack.append((v, u))
        ys, xs = np.array(pts).T
        blobs.append((float(m[ys, xs].sum()), xs.min(), ys.min(), xs.max(), ys.max()))
    blobs.sort(reverse=True)
    return blobs[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--durations", default="1s,3s,7s")
    ap.add_argument("--regions")
    ap.add_argument("--out", default="attention-out")
    a = ap.parse_args()
    durs = [d.strip() for d in a.durations.split(",") if d.strip()]
    for d in durs:
        if not (MODELS / f"foveacast-v3-{d}-int8.onnx").exists():
            sys.exit(f"no model for duration {d}; choose from 1s, 3s, 7s")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sessions = {d: ort.InferenceSession(str(MODELS / f"foveacast-v3-{d}-int8.onnx"), opts, providers=["CPUExecutionProvider"]) for d in durs}
    regions = json.loads(Path(a.regions).read_text()) if a.regions else []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for path in a.images:
        img = Image.open(path).convert("RGB")
        ow, oh = img.size
        sx, sy = ow / W, oh / H
        lum = float(np.asarray(img.convert("L"), np.float32).mean() / 255)
        entry = {"image": path, "size": [ow, oh], "durations": {}, "notes": []}
        if lum < 0.35:
            entry["notes"].append(f"dark UI (mean luminance {lum:.2f}): the model was trained mostly on light UIs; treat results with care")
        if max(ow, oh) / min(ow, oh) > 2.2:
            entry["notes"].append("very tall/wide image is squashed to 320x240; crop to one viewport for better results")
        tiles = []
        for d, sess in sessions.items():
            m = predict(sess, img)
            total = float(m.sum()) or 1.0
            norm = m / (m.max() or 1.0)
            heat = Image.fromarray(colormap(norm).astype(np.uint8)).resize((ow, oh), Image.BILINEAR)
            alpha = Image.fromarray((np.clip(norm, 0, 1) ** 0.8 * 190).astype(np.uint8)).resize((ow, oh), Image.BILINEAR)
            over = Image.composite(heat, img, alpha)  # heat shows only where attention is; quiet areas keep the UI
            over.save(out / f"{Path(path).stem}-{d}.png")
            tiles.append(over)
            spots = [{"share": round(s / total, 3), "box": [round(x0 * sx), round(y0 * sy), round((x1 - x0 + 1) * sx), round((y1 - y0 + 1) * sy)]}
                     for s, x0, y0, x1, y1 in hotspots(m)]
            regs = []
            for r in regions:
                x, y, w, h = r["box"]
                x0, y0 = int(x / sx), int(y / sy)
                x1, y1 = max(x0 + 1, int(np.ceil((x + w) / sx))), max(y0 + 1, int(np.ceil((y + h) / sy)))
                share = float(m[y0:y1, x0:x1].sum()) / total
                area = (x1 - x0) * (y1 - y0) / (W * H)
                regs.append({"name": r["name"], "share": round(share, 3), "vs_area": round(share / area, 2) if area else None})
            regs.sort(key=lambda r: -r["share"])
            for i, r in enumerate(regs, 1):
                r["rank"] = i
            entry["durations"][d] = {"hotspots": spots, "regions": regs}
        strip = Image.new("RGB", (ow * len(tiles), oh))
        for i, t in enumerate(tiles):
            strip.paste(t, (i * ow, 0))
        strip.save(out / f"{Path(path).stem}-strip.png")
        (out / f"{Path(path).stem}.json").write_text(json.dumps(entry, indent=1))
        report.append(entry)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
