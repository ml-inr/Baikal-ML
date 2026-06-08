"""Score distribution histograms per particle type (data_class).

Usage (from project root):
    python inference_v2/nu_classifier/analysis/score_dist_by_class.py \\
        --preds-dir inference_v2/nu_classifier/preds/260508_1724_...@best_da_model \\
        [--source mc_merged] [--thr 0.8] [--bins 80] [--log-y] \\
        [--save inference_v2/nu_classifier/analysis/score_dist.png]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from inference_v2.nu_classifier.analysis.load import load_preds

CLASS_COLORS = {
    "muatm_2020":  "tab:blue",
    "nuatm_2020":  "tab:orange",
    "nue2_2020":   "tab:green",
    "muatm":       "tab:blue",
    "nuatm_conv":  "tab:orange",
    "nue2":        "tab:green",
}
CLASS_ORDER = ["muatm_2020", "nuatm_2020", "nue2_2020", "muatm", "nuatm_conv", "nue2"]


def plot_score_dist_by_class(
    df,
    n_bins: int = 80,
    log_y: bool = True,
    save_path: str | None = None,
):
    classes = [c for c in CLASS_ORDER if c in df["data_class"].unique()]
    classes += sorted(set(df["data_class"].unique()) - set(CLASS_ORDER))

    bins = np.linspace(0, 1, n_bins + 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls in classes:
        scores = df.loc[df["data_class"] == cls, "score"].values
        color = CLASS_COLORS.get(cls)
        ax.hist(
            scores,
            bins=bins,
            histtype="step",
            density=True,
            label=f"{cls}  (n={len(scores):,})",
            color=color,
            linewidth=1.4,
        )

    ax.set_xlabel("Nu-classifier score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score distribution by particle type")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds-dir", required=True,
                        help="Path to preds/{checkpoint_name}/ directory")
    parser.add_argument("--source",    default="mc_merged")
    parser.add_argument("--thr",       type=float, default=0.8)
    parser.add_argument("--catalog",   default="data_manager/catalog_v2.duckdb")
    parser.add_argument("--bins",      type=int,   default=80)
    parser.add_argument("--log-y",     action="store_true", default=True)
    parser.add_argument("--no-log-y",  dest="log_y", action="store_false")
    parser.add_argument("--save",      default=None,
                        help="Output image path (e.g. score_dist.png). "
                             "Defaults to {preds_dir}/score_dist_by_class.png")
    args = parser.parse_args()

    save_path = args.save or str(Path(args.preds_dir) / "score_dist_by_class.png")

    print(f"Loading predictions from {args.preds_dir} ...")
    df = load_preds(
        args.preds_dir,
        source=args.source,
        thr=args.thr,
        catalog_path=args.catalog,
    )
    print(f"  {len(df):,} events  |  classes: {df['data_class'].value_counts().to_dict()}")

    plot_score_dist_by_class(df, n_bins=args.bins, log_y=args.log_y, save_path=save_path)
    plt.show()


if __name__ == "__main__":
    main()
