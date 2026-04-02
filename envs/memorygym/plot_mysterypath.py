import json
import re
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


def parse_arguments():
    parser = ArgumentParser(description="Plot survival rate overlay for multiple context lengths")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to step directory (e.g. eval_results/.../step_1500000000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: results-dir/overall_graph.png)")
    parser.add_argument("--thresholds", type=int, nargs="+",
                        default=list(range(512, 1537, 32)),
                        help="Step thresholds for x-axis")
    return parser.parse_args()


def discover_ctx_results(results_dir: Path):
    """Find all *_ctx* subdirs with batch_results.json, return sorted by ctx length."""
    entries = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        match = re.search(r"_ctx(\d+)$", subdir.name)
        if match is None:
            continue
        results_file = subdir / "batch_results.json"
        if not results_file.exists():
            continue
        entries.append((int(match.group(1)), results_file))
    entries.sort(key=lambda x: x[0])
    return entries


def main():
    args = parse_arguments()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    entries = discover_ctx_results(results_dir)
    if not entries:
        raise FileNotFoundError(f"No *_ctx*/batch_results.json found in {results_dir}")

    thresholds = sorted(args.thresholds)
    colors = sns.color_palette("deep", n_colors=len(entries))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for (ctx_len, results_file), color in zip(entries, colors):
        with open(results_file) as f:
            data = json.load(f)

        lengths = [ep["length"] for ep in data["episodes"]]
        total = len(lengths)
        survival_rates = [sum(l >= t for l in lengths) / total for t in thresholds]

        ax.plot(thresholds, survival_rates, marker="o", linewidth=2.5, markersize=6,
                color=color, label=f"ctx={ctx_len}", zorder=5)

    major_ticks = [t for t in thresholds if t % 128 == 0]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(t) for t in major_ticks], fontsize=11, fontweight="bold")
    ax.set_xlabel("Step Threshold", fontsize=12)
    ax.set_ylabel("Survival Rate", fontsize=12)

    env_name = data.get("env_name", "Endless-MysteryPath-v0")
    ax.set_title(f"Survival Rate vs Step Threshold ({env_name})",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(-0.02, 1.12)
    ax.legend(fontsize=11, loc="upper right")

    fig.tight_layout()

    if args.output:
        out_path = args.output
    else:
        out_path = str(results_dir / "overall_graph.png")

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
