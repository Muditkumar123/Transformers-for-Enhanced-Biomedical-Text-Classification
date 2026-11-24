"""Build tables and figures (paper Fig. 4, 5, 6 and Table 1) from results/<dataset>/model_*.json."""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["20k", "200k"], default="20k")
ap.add_argument("--framework", choices=["tf", "torch"], default="tf")
args = ap.parse_args()
OUT = Path(__file__).resolve().parent / "results" / args.framework / args.dataset
runs = sorted((json.loads(p.read_text()) for p in OUT.glob("model_*.json")), key=lambda r: r["model"])
assert runs, f"no results in {OUT}"

lines = [f"# Results: {args.framework}, PubMed {args.dataset} RCT\n"]
for split in ("val", "test"):
    df = pd.DataFrame([{"model": r["model"], "name": r["name"], **r[split],
                        "train_min": r["train_seconds"] / 60} for r in runs]).set_index("model")
    df["accuracy"] = df["accuracy"] / 100
    df.to_csv(OUT / f"summary_{split}.csv", float_format="%.4f")
    lines += [f"## {split} split (weighted precision/recall/F1)\n", df.to_markdown(floatfmt=".4f"), ""]
    if split == "val":
        sorted_f1 = df.set_index("name")["f1"].sort_values(ascending=False)
        sorted_f1.plot(kind="bar", figsize=(10, 7), title=f"F1 by model ({args.framework}, val, {args.dataset})")
        plt.tight_layout(); plt.savefig(OUT / "fig4_f1_comparison.png", dpi=120); plt.close()
        df.set_index("name")[["accuracy", "precision", "recall", "f1"]].plot(kind="bar", figsize=(10, 7))
        plt.legend(bbox_to_anchor=(1.0, 1.0)); plt.tight_layout()
        plt.savefig(OUT / "fig5_metrics_comparison.png", dpi=120); plt.close()

m6 = next((r for r in runs if r["model"] == 6), None)
if m6:
    classes = [k for k in m6["val_report"] if k not in ("accuracy", "macro avg", "weighted avg")]
    paper = {"BACKGROUND": 0.77, "CONCLUSIONS": 0.89, "METHODS": 0.91, "OBJECTIVE": 0.63, "RESULTS": 0.90}
    t1 = pd.DataFrame({"paper F1": [paper[c] for c in classes] + [0.87],
                       "ours val F1": [m6["val_report"][c]["f1-score"] for c in classes] + [m6["val_report"]["accuracy"]],
                       "ours test F1": [m6["test_report"][c]["f1-score"] for c in classes] + [m6["test_report"]["accuracy"]]},
                      index=classes + ["ACCURACY"])
    lines += ["## Table 1 (Model 6 per-class F1) vs paper\n", t1.to_markdown(floatfmt=".2f"), ""]
    cm = np.array(m6["val_confusion"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)), classes, rotation=30); ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label"); ax.set_title(f"Model 6 confusion matrix ({args.framework}, val, {args.dataset})")
    fig.colorbar(im); plt.tight_layout(); plt.savefig(OUT / "fig6_confusion_matrix_model6.png", dpi=120); plt.close()

(OUT / "RESULTS.md").write_text("\n".join(lines))
print("\n".join(lines))
