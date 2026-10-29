"""Training curves of GRPO for the three base models (Fig. 3 of the OJ-CS article).

Reads ckpt/<run>/log.jsonl (one line per GRPO step: mean sampled quality ratio 'qr' and valid fraction 'valid')
and draws, for each base model, the mean over the seeds (line) and the range over seeds (band) for the
named and the property-based inventory; top row quality ratio, bottom row validity. A seed given with --mark is
also drawn as its own dotted line; it stays in the mean and band of its configuration.
Bands are drawn in a lightened opaque colour, so the PDF has no transparency.

    python3 -m cuas_rlvr.figure_curves --ckpt ckpt --mark q38_27banon:0 --out figures/training_curves_all.pdf
"""
import argparse
import json
import os

import numpy as np

MODELS = [("Qwen2.5-7B", "v2_q7b", "v2anon_q7b"), ("Qwen3-8B", "q3_8b", "q3_8banon"), ("Qwen3.8-27B", "q38_27b", "q38_27banon")]
# Okabe-Ito colour-blind-safe pair, distinguished also by line style
STYLE = {"names": dict(color="#0072B2", ls="-"), "properties": dict(color="#D55E00", ls="--")}


def smooth(y, w):
    return np.convolve(y, np.ones(w) / w, mode="valid")


def light(hexcol, f=0.78):
    c = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(c + (1 - c) * f)


def load(ckpt, tag, key, w, seeds=(0, 1, 2)):
    runs = []
    for s in seeds:
        f = os.path.join(ckpt, f"{tag}_grpo_s{s}", "log.jsonl")
        if os.path.exists(f):
            runs.append(smooth(np.array([json.loads(l)[key] for l in open(f) if l.strip()]), w))
    n = min(len(r) for r in runs)
    return np.vstack([r[:n] for r in runs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--out", default="training_curves_all.pdf")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--mark", action="append", default=[], help="tag:seed also drawn alone (dotted), e.g. q38_27banon:0")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"], "font.size": 8, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.5), dpi=300, sharex=True, sharey="row",
                             gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.12, wspace=0.08))
    for j, (title, tag, atag) in enumerate(MODELS):
        for row, key in enumerate(("qr", "valid")):
            ax = axes[row, j]
            for lab, t in (("names", tag), ("properties", atag)):
                marked = [int(x.split(":")[1]) for x in a.mark if x.split(":")[0] == t]
                Y = load(a.ckpt, t, key, a.window)
                x = np.arange(a.window, a.window + Y.shape[1])
                st = STYLE[lab]
                ax.fill_between(x, Y.min(0), Y.max(0), color=light(st["color"]), lw=0, zorder=1)
                ax.plot(x, Y.mean(0), color=st["color"], ls=st["ls"], lw=1.4, zorder=3,
                        label=f"{lab}: mean of 3 seeds (band: min–max)")
                for s in marked:
                    Z = load(a.ckpt, t, key, a.window, (s,))[0]
                    ax.plot(np.arange(a.window, a.window + len(Z)), Z, color=st["color"], ls=":", lw=1.0, zorder=2,
                            label=f"{lab}, seed {s}")
            ax.grid(axis="y", alpha=0.3, lw=0.4)
            ax.set_xlim(0, 300)
            ax.set_xticks([0, 100, 200, 300])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(length=2.5, width=0.5, labelsize=7.5)
        axes[0, j].set_title(title, fontsize=9)
        axes[1, j].set_xlabel("GRPO step")
    axes[0, 0].set_ylabel("Sampled quality ratio")
    axes[1, 0].set_ylabel("Valid fraction")
    axes[0, 0].set_ylim(0.5, 1.0)
    axes[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[1, 0].set_ylim(0.7, 1.01)
    axes[1, 0].set_yticks([0.7, 0.8, 0.9, 1.0])
    hs = {}
    for ax in axes[0]:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            hs.setdefault(l_, h_)
    h, l = list(hs.values()), list(hs.keys())
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01), handlelength=3)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.2)
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
