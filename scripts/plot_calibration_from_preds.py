"""Reliability curve and ECE (5 equal-count bins) from a predictions.csv with label, prob.

  python scripts/plot_calibration_from_preds.py <predictions.csv> <out.png> <title>
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out, title = sys.argv[1:4]
d = pd.read_csv(src); y, p = d.label.values.astype(int), d.prob.values
NB = 5
b = np.clip(np.digitize(p, np.quantile(p, np.linspace(0, 1, NB + 1)[1:-1])), 0, NB - 1)
conf = np.array([p[b == i].mean() for i in range(NB) if (b == i).any()])
acc = np.array([y[b == i].mean() for i in range(NB) if (b == i).any()])
w = np.array([(b == i).mean() for i in range(NB) if (b == i).any()])
ece = float(np.sum(w * np.abs(acc - conf)))
fig, ax = plt.subplots(figsize=(4.4, 4.2))
ax.plot([0, 1], [0, 1], "--", color="0.7", lw=0.8)
ax.plot(conf, acc, "-o", color="#d62728", ms=5, lw=1.5)
ax.text(0.04, 0.93, f"ECE = {ece:.3f}", transform=ax.transAxes, fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("predicted probability"); ax.set_ylabel("observed progression rate")
ax.set_title(f"{title}\n(n={len(p)})", fontsize=8.5)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(out, dpi=200)
print(f"{title}: n={len(p)} pos={y.sum()} ECE={ece:.3f} mean_pred={p.mean():.3f} event_rate={y.mean():.3f}")
