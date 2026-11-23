"""Regenerate the README figures from the committed evidence in results/ and assets/powai/.

    python scripts/make_figures.py

Every number plotted here is read from a JSON file in results/ or from the paper constants below
(which are labelled as published values). Nothing is typed in by hand except the paper's reported F1 scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures"

# Published LEVIR-CD test F1 (Codegoni, Lombardi & Ferrari 2023, Table 4 and Table 3).
PAPER_F1 = {"tinycd": 91.05, "bit": 89.31, "changeformer": 90.40}
LABELS = {"tinycd": "TinyCD", "bit": "BIT", "changeformer": "ChangeFormer V6"}
COLORS = {"tinycd": "#1f6feb", "bit": "#8b949e", "changeformer": "#d29922"}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def fig_f1_benchmark(stats: dict) -> None:
    keys = ["tinycd", "changeformer", "bit"]
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    y = np.arange(len(keys))
    f1 = [stats["ci"][k]["f1"] for k in keys]
    stats_hi = [stats["ci"][k]["high"] for k in keys]
    lo = [stats["ci"][k]["f1"] - stats["ci"][k]["low"] for k in keys]
    hi = [stats["ci"][k]["high"] - stats["ci"][k]["f1"] for k in keys]
    ax.barh(y, f1, color=[COLORS[k] for k in keys], height=0.55, alpha=0.9,
            label="Measured (this replication)")
    ax.errorbar(f1, y, xerr=[lo, hi], fmt="none", ecolor="black", capsize=4, lw=1.2,
                label="95% paired bootstrap CI")
    ax.scatter([PAPER_F1[k] for k in keys], y, marker="D", color="white", edgecolor="black", zorder=5,
               label="Published (paper)")
    for yi, k, v, h in zip(y, keys, f1, stats_hi):
        ax.text(h + 0.08, yi, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, [LABELS[k] for k in keys])
    ax.set_xlim(88.5, 92.2)
    ax.set_xlabel("F1 of the change class (%), LEVIR-CD test set, 2,048 pairs, threshold 0.5")
    ax.set_title("Released checkpoints, frozen inference: F1 on LEVIR-CD test", fontsize=11, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "01_benchmark_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_object_recall(models: dict[str, dict]) -> None:
    keys = ["tinycd", "bit", "changeformer"]
    bins = models["tinycd"]["objects"]["bins"]
    names = [b["label"] for b in bins]
    x = np.arange(len(bins))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, k in enumerate(keys):
        vals = [b["recall_pct"] for b in models[k]["objects"]["bins"]]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=COLORS[k], label=LABELS[k])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("Object recall (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Object-level recall by building footprint size (test set)", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "02_object_recall_by_size.png", dpi=150)
    plt.close(fig)


def fig_efficiency(eff: dict) -> None:
    keys = ["tinycd", "bit", "changeformer"]
    panels = [
        ("Parameters used in forward pass (M)", [eff["models"][k]["params_used"] / 1e6 for k in keys], "{:.2f}"),
        ("Compute (GFLOPs, 256x256 pair)", [eff["models"][k]["gflops"] for k in keys], "{:.1f}"),
        ("GPU latency, median (ms, batch 1)", [eff["models"][k]["latency_gpu_ms"]["median"] for k in keys], "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (title, vals, fmt) in zip(axes, panels):
        bars = ax.bar([LABELS[k] for k in keys], vals, color=[COLORS[k] for k in keys])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02, fmt.format(v), ha="center", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=12)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cost of the released models (RTX 3050 6 GB laptop GPU, PyTorch 2.7.0)", fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "03_efficiency.png", dpi=150)
    plt.close(fig)


def fig_probes(probes: dict) -> None:
    order = ["none", "no_M64", "no_M128", "no_M256", "no_masks"]
    names = {"none": "Unmodified", "no_M64": "M64 set to 1", "no_M128": "M128 set to 1",
             "no_M256": "M256 set to 1", "no_masks": "All masks set to 1"}
    vals = [probes["probes"][k]["metrics"]["f1"] for k in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    y = np.arange(len(order))[::-1]
    bars = ax.barh(y, vals, color=["#1f6feb"] + ["#f85149"] * (len(order) - 1), height=0.55)
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(y, [names[k] for k in order], fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("F1 of the change class (%) on the test set")
    ax.set_title("Inference-time probe: set attention masks of the released TinyCD to ones", fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "04_mask_probe.png", dpi=150)
    plt.close(fig)


def fig_powai() -> None:
    pair = ROOT / "assets" / "powai" / "pair"
    a = Image.open(pair / "A.png").convert("RGB")
    b = Image.open(pair / "B.png").convert("RGB")
    v = Image.open(pair / "valid.png").convert("L")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    axes[0].imshow(a)
    axes[0].set_title("t1: 10 Feb 2015 (Landsat 8, 30 m)", fontsize=9)
    axes[1].imshow(b)
    axes[1].set_title("t2: 5 Feb 2025 (Landsat 8, 30 m)", fontsize=9)
    axes[2].imshow(np.array(v), cmap="gray")
    axes[2].set_title("Valid pixels in both dates (99.80%)", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("IIT Bombay / Powai crop, 256 x 256 px (7.7 km). Credit: U.S. Geological Survey Landsat 8 "
                 "Collection 2 Level-2", fontsize=8, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "05_powai_input_pair.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    stats = load("stats.json")
    models = {k: load(f"eval_{k}_test.json") for k in ["tinycd", "bit", "changeformer"]}
    fig_f1_benchmark(stats)
    fig_object_recall(models)
    fig_efficiency(load("efficiency.json"))
    fig_probes(load("probes.json"))
    fig_powai()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
