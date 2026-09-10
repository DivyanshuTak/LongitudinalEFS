"""AUROC by landmark year from one eval_landmarks.py metrics.txt, with 95% CI and N.

  python scripts/plot_landmark_single.py <metrics.txt> <out.png> <color> [title]
"""
import re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse(path):
    pts = []
    for ln in open(path):
        m = re.match(r"\s*(?:year )?(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+) \[([\d.]+), ([\d.]+)\]", ln)
        if m and m.group(1) != "pooled":
            pts.append((m.group(1), int(m.group(2)), float(m.group(4)), float(m.group(5)), float(m.group(6))))
    return pts

src, out, color = sys.argv[1:4]
title = sys.argv[4] if len(sys.argv) > 4 else None
pts = parse(src); x = list(range(len(pts))); y = [p[2] for p in pts]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(x, y, yerr=[[p[2] - p[3] for p in pts], [p[4] - p[2] for p in pts]], fmt="-o", color=color, ms=5, lw=1.5, capsize=3)
for xi, yi, p in zip(x, y, pts):
    ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(6, 4), textcoords="offset points", fontsize=8, color=color)
    ax.annotate(f"N={p[1]}", (xi, 0.13), ha="center", fontsize=8, color="0.4")
ax.set_xticks(x, [p[0] for p in pts]); ax.axhline(0.5, color="0.7", lw=0.8, ls="--")
ax.set_xlabel("landmark year"); ax.set_ylabel("AUROC"); ax.set_ylim(0.1, 1.02)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
if title: ax.set_title(title, fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=200)
