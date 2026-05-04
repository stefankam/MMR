from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "figure.figsize": (8.0, 5.0),
    "figure.dpi": 200,
    "savefig.dpi": 300,

    "font.family": "serif",
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "legend.fontsize": 13,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    "axes.linewidth": 1.3,
    "lines.linewidth": 2.8,
    "lines.markersize": 7,

    "grid.linewidth": 0.8,
    "grid.alpha": 0.28,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Load
# ============================================================
df = pd.read_csv("results.csv")

# Optional output directory
outdir = Path("paper_plots")
outdir.mkdir(exist_ok=True)

# ============================================================
# Helper: robust flag parser
# ============================================================
def has_flag_value(v):
    s = str(v).strip()
    return s not in {"", "[]", "nan", "None"}

# ============================================================
# Core metric function
# This keeps your round-aggregated logic exactly in spirit.
# ============================================================
def compute_summary_metrics(df_slice):
    df_slice = df_slice.sort_values("round").copy()

    # Aggregate per round
    round_scores = df_slice.groupby("round")["score"].mean().to_dict()
    round_flags = df_slice.groupby("round")["flags"].apply(
        lambda x: any(has_flag_value(v) for v in x)
    ).to_dict()
    attack_active = df_slice.groupby("round")["y_true"].max().to_dict()

    rounds_sorted = sorted(round_scores.keys())

    y = [1 if attack_active[r] else 0 for r in rounds_sorted]
    s = [round_scores[r] for r in rounds_sorted]
    alarm = [1.0 if round_flags.get(r, False) else 0.0 for r in rounds_sorted]

    def auc_rank(y_true, scores):
        pos = [ss for yy, ss in zip(y_true, scores) if yy == 1]
        neg = [ss for yy, ss in zip(y_true, scores) if yy == 0]
        if not pos or not neg:
            return np.nan

        wins = 0.0
        for p in pos:
            for n in neg:
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
        return wins / (len(pos) * len(neg))

    score_auc = auc_rank(y, s)
    alarm_auc = auc_rank(y, alarm)

    auc_terms = [x for x in [score_auc, alarm_auc] if not np.isnan(x)]
    auc = float(np.mean(auc_terms)) if auc_terms else np.nan

    # TTD: latest attack episode only
    TTD = np.nan
    if rounds_sorted:
        attack_starts = []
        prev = False

        for r in rounds_sorted:
            cur = bool(attack_active.get(r, False))
            if cur and not prev:
                attack_starts.append(r)
            prev = cur

        if attack_starts:
            latest_start = max(attack_starts)
            current_r = rounds_sorted[-1]

            detected = any(
                round_flags.get(r, False)
                for r in rounds_sorted
                if latest_start <= r <= current_r
            )

            TTD = 0.0 if detected else float(current_r - latest_start)

    return {"AUC": auc, "TTD": TTD}

# ============================================================
# Running summary over rounds for each detector/seed/q/p
# ============================================================
def compute_running_summary(df):
    rows = []

    group_cols = ["detector", "seed", "q", "p"]

    for keys, sub in df.groupby(group_cols):
        det, seed, q, p = keys
        sub = sub.sort_values("round").copy()
        rounds = sorted(sub["round"].unique())

        for r in rounds:
            sub_until_r = sub[sub["round"] <= r]
            m = compute_summary_metrics(sub_until_r)

            rows.append({
                "detector": det,
                "seed": seed,
                "q": q,
                "p": p,
                "round": r,
                "AUC": m["AUC"],
                "TTD": m["TTD"],
            })

    return pd.DataFrame(rows)

summary_round_df = compute_running_summary(df)
print(summary_round_df.head())
print(summary_round_df.shape)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)

    ax.grid(True, alpha=0.28, linewidth=0.8)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=14,
        width=1.3,
        length=5
    )






def aggregate_over_seeds(summary_df):
    agg = (
        summary_df
        .groupby(["detector", "q", "p", "round"], as_index=False)
        .agg(
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", "std"),
            TTD_mean=("TTD", "mean"),
            TTD_std=("TTD", "std"),
            n=("seed", "count"),
        )
    )
    return agg

agg_df = aggregate_over_seeds(summary_round_df)
print(agg_df.head())

def plot_aggregated_metric(agg_df, metric_mean, metric_std, ylabel, save_prefix):
    for (q, p), sub in agg_df.groupby(["q", "p"]):
        fig, ax = plt.subplots(figsize=(8, 5))

        for det, det_df in sub.groupby("detector"):
            det_df = det_df.sort_values("round")

            x = det_df["round"].to_numpy()
            y = det_df[metric_mean].to_numpy(dtype=float)
            sd = det_df[metric_std].fillna(0.0).to_numpy(dtype=float)

            ax.plot(x, y, marker="o", linewidth=2.0, markersize=4, label=det)
            ax.fill_between(x, y - sd, y + sd, alpha=0.15)

        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Round | q={q}, p={p}")
        if ylabel == "AUC":
            ax.set_ylim(0, 1.05)
        style_axes(ax)
        ax.legend(frameon=False)
        plt.tight_layout()

        fname = outdir / f"{save_prefix}_q{q}_p{p}.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.show()
        plt.close()

plot_aggregated_metric(
    agg_df,
    metric_mean="AUC_mean",
    metric_std="AUC_std",
    ylabel="AUC",
    save_prefix="auc_mean_std_vs_round",
)

plot_aggregated_metric(
    agg_df,
    metric_mean="TTD_mean",
    metric_std="TTD_std",
    ylabel="TTD",
    save_prefix="ttd_mean_std_vs_round",
)



def plot_metric_seeds_per_detector(summary_df, metric="AUC", q_value=None, p_value=None):
    sub = summary_df.copy()
    if q_value is not None:
        sub = sub[sub["q"] == q_value]
    if p_value is not None:
        sub = sub[sub["p"] == p_value]

    for det, det_df in sub.groupby("detector"):
        fig, ax = plt.subplots(figsize=(6.5, 4.0))

        for seed, seed_df in det_df.groupby("seed"):
            seed_df = seed_df.sort_values("round")
            ax.plot(
                seed_df["round"],
                seed_df[metric],
                marker="o",
                linewidth=1.5,
                markersize=3.5,
                label=f"seed={seed}",
            )

        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        title = f"{det}: {metric} vs Round"
        if q_value is not None:
            title += f" | q={q_value}"
        if p_value is not None:
            title += f", p={p_value}"
        ax.set_title(title)

        if metric == "AUC":
            ax.set_ylim(0, 1.05)

        style_axes(ax)
        ax.legend(frameon=False, ncol=2, fontsize=13)
        plt.tight_layout()

        fname = outdir / f"{metric.lower()}_{det}_seeds"
        if q_value is not None:
            fname = str(fname) + f"_q{q_value}"
        if p_value is not None:
            fname = str(fname) + f"_p{p_value}"
        fname = fname + ".pdf"

        plt.savefig(fname, bbox_inches="tight")
        plt.show()
        plt.close()

# Example:
# plot_metric_seeds_per_detector(summary_round_df, metric="AUC", q_value=0.2, p_value=0.2)
# plot_metric_seeds_per_detector(summary_round_df, metric="TTD", q_value=0.2, p_value=0.2)


def plot_mean_auc_vs_q(summary_round_df, outdir=None, filename="mean_auc_vs_q.png"):
    """
    Plot mean final AUC vs q for each detector.

    Parameters
    ----------
    summary_round_df : pd.DataFrame
        Output of compute_running_summary(df), with columns:
        detector, seed, q, p, round, AUC, TTD
    outdir : pathlib.Path or str, optional
        Directory to save the figure. If None, only shows the plot.
    filename : str
        Output filename if outdir is provided.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Keep only valid AUC rows
    plot_df = summary_round_df.dropna(subset=["AUC"]).copy()

    if plot_df.empty:
        print("No valid AUC values found in summary_round_df.")
        return

    # ------------------------------------------------------------
    # For each detector/seed/q/p, keep the FINAL running summary
    # ------------------------------------------------------------
    final_df = (
        plot_df.sort_values("round")
               .groupby(["detector", "seed", "q", "p"], as_index=False)
               .tail(1)
               .copy()
    )

    # ------------------------------------------------------------
    # Average final AUC across seeds (and across p if multiple p's exist)
    # ------------------------------------------------------------
    mean_df = (
        final_df.groupby(["detector", "q"], as_index=False)
                .agg(
                    mean_auc=("AUC", "mean"),
                    std_auc=("AUC", "std"),
                    n=("AUC", "count")
                )
                .sort_values(["detector", "q"])
    )

    # Replace NaN std when only one seed exists
    mean_df["std_auc"] = mean_df["std_auc"].fillna(0.0)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    for detector, sub in mean_df.groupby("detector"):
        sub = sub.sort_values("q")
        ax.plot(sub["q"], sub["mean_auc"], marker="o", label=detector)
        ax.fill_between(
            sub["q"],
            sub["mean_auc"] - sub["std_auc"],
            sub["mean_auc"] + sub["std_auc"],
            alpha=0.15
        )

    ax.set_xlabel("Availability q")
    ax.set_ylabel("Mean final AUC")
    ax.set_title("Mean Final AUC vs q")
    ax.set_ylim(0.0, 1.05)
    style_axes(ax)
    ax.legend(frameon=False)

    plt.tight_layout()

    if outdir is not None:
        from pathlib import Path
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")

    plt.show()

plot_mean_auc_vs_q(summary_round_df, outdir=outdir)



def aggregate_auc_over_seeds_by_attack(summary_df, original_df, detector_value, q_value, p_value):
    # Restrict to chosen detector / q / p
    base = summary_df[
        (summary_df["detector"] == detector_value) &
        (summary_df["q"] == q_value) &
        (summary_df["p"] == p_value)
    ].copy()

    # Attach attack labels from the original round-level data
    attack_map = (
        original_df[["seed", "detector", "q", "p", "round", "attack"]]
        .drop_duplicates()
        .copy()
    )

    base = base.merge(
        attack_map,
        on=["seed", "detector", "q", "p", "round"],
        how="left"
    )

    # Aggregate by attack and round
    agg = (
        base
        .groupby(["attack", "round"], as_index=False)
        .agg(
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", lambda x: x.std(ddof=1)),
            n_valid=("AUC", lambda x: int(x.notna().sum())),
        )
        .sort_values(["attack", "round"])
        .reset_index(drop=True)
    )
    print(agg)
    return agg


def plot_auc_vs_round_by_attack(summary_df, original_df, detector_value, q_value, p_value):
    agg = aggregate_auc_over_seeds_by_attack(
        summary_df=summary_df,
        original_df=original_df,
        detector_value=detector_value,
        q_value=q_value,
        p_value=p_value,
    )

    if agg.empty:
        print(f"No data found for detector={detector_value}, q={q_value}, p={p_value}")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    attack_order = ["backdoor", "scaled", "slow_drift"]
    attack_styles = {
        "backdoor":   {"linestyle": "-",  "marker": "o", "zorder": 3},
        "scaled":     {"linestyle": "--", "marker": "s", "zorder": 4},
        "slow_drift": {"linestyle": ":",  "marker": "^", "zorder": 2},
    }

    for attack in attack_order:
        sub = agg[agg["attack"] == attack].sort_values("round").copy()
        if sub.empty:
            print(f"Skipping {attack}: no rows found")
            continue

        print(f"\nATTACK: {attack}")
        print(sub[["round", "AUC_mean", "AUC_std", "n_valid"]].head(20))

        x = sub["round"].to_numpy(dtype=float)
        y = sub["AUC_mean"].to_numpy(dtype=float)
        sd = sub["AUC_std"].to_numpy(dtype=float)
        n = sub["n_valid"].to_numpy(dtype=int)

        # Safe CI computation: only where at least 2 valid seeds exist
        ci = np.full(y.shape, np.nan, dtype=float)
        valid = (~np.isnan(y)) & (~np.isnan(sd)) & (n >= 2)
        ci[valid] = 1.96 * sd[valid] / np.sqrt(n[valid])

        style = attack_styles.get(attack, {})

        ax.plot(
            x,
            y,
            label=attack,
            linewidth=2.8,
            markersize=7,
            alpha=0.95,
            **style
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            where=~np.isnan(ci),
            alpha=0.12,
            interpolate=False
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("AUC")
    ax.set_title(f"AUC vs Round by Attack | detector={detector_value}, q={q_value}, p={p_value}")
    ax.set_ylim(0, 1.05)
    style_axes(ax)
    ax.legend(frameon=False)

    plt.tight_layout()
    fname = outdir / f"auc_vs_round_by_attack_{detector_value}_q{q_value}_p{p_value}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.show()
    plt.close()


plot_auc_vs_round_by_attack(
    summary_df=summary_round_df,
    original_df=df,
    detector_value="MMR",
    q_value=0.2,
    p_value=0.2,
)