import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid.axis": "y",
})

FIGSIZE = (11, 6)
DPI = 150
COLOR = sns.color_palette("deep")[0]


def parse_arguments():
    parser = ArgumentParser(description="Plot survival rate for MysteryPath batch eval")
    parser.add_argument("--results-path", type=str, required=True, help="Path to batch_results.json")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (default: same dir as results)")
    parser.add_argument("--thresholds", type=int, nargs="+",
                        default=list(range(512, 1537, 32)),
                        help="Step thresholds for x-axis")
    return parser.parse_args()


def main():
    args = parse_arguments()
    with open(args.results_path) as f:
        data = json.load(f)

    lengths = [ep["length"] for ep in data["episodes"]]
    total = len(lengths)

    thresholds = sorted(args.thresholds)
    survival_rates = [sum(l >= t for l in lengths) / total for t in thresholds]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(thresholds, survival_rates, marker="o", linewidth=2.5, markersize=8,
            color=COLOR, zorder=5)

    major_ticks = [t for t in thresholds if t % 128 == 0]
    for t, sr in zip(thresholds, survival_rates):
        if t in major_ticks:
            ax.annotate(f"{sr:.0%}", (t, sr), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=10, color=COLOR,
                        fontweight="bold")

    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(t) for t in major_ticks], fontsize=11, fontweight="bold")
    ax.set_xlabel("Step Threshold", fontsize=12)
    ax.set_ylabel("Survival Rate", fontsize=12)

    env_name = data.get("env_name", "Endless-MysteryPath-v0")
    num_episodes = data.get("num_episodes", total)
    ax.set_title(f"Survival Rate vs Step Threshold ({env_name}, n={num_episodes})",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(-0.02, 1.12)

    fig.tight_layout()

    if args.output:
        out_path = args.output
    else:
        out_path = str(Path(args.results_path).parent / "mysterypath_survival.png")

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
