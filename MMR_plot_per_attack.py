from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.6,
    "lines.markersize": 6,
})

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = "detector_results_p0.2.csv"
OUTDIR = Path("paper_plots")
OUTDIR.mkdir(exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def load_results(path: str) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

    # Clean BOM / whitespace
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    numeric_cols = [
        "seed", "Nm", "q", "p", "round",
        "y_true", "score", "fpr", "summary_auc", "summary_ttd"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "detector" in df.columns:
        df["detector"] = df["detector"].astype(str).str.strip().str.upper()

    if "attack" in df.columns:
        df["attack"] = df["attack"].astype(str).str.strip().str.lower()

    return df


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.grid(True, alpha=0.25, linewidth=0.8)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=14,
        width=1.2,
        length=5
    )




def first_detection_round(group: pd.DataFrame):
    detected = group.loc[group["summary_auc"] > 0, "round"]
    return detected.min() if len(detected) else np.nan


# ============================================================
# Step 1: collapse repeated rows within each seed/round
# ============================================================
def aggregate_round_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate rows within each
    (seed, detector, Nm, q, p, attack, round).

    This is needed because Nm=3 produces 3 rows per round.
    """
    df = df.copy()
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    keys = ["seed", "detector", "Nm", "q", "p", "attack", "round"]

    agg = (
        df.groupby(keys, as_index=False)
          .agg({
              "y_true": "max",          # any malicious event in that seed/round
              "score": "max",           # strongest anomaly score in that round
              "fpr": "max",             # conservative round-level FPR
              "summary_auc": "max",     # keep the round summary statistic
              "summary_ttd": "min",     # earliest TTD seen in that round
          })
    )

    return agg


# ============================================================
# Step 2: average across seeds for plotting curves
# ============================================================
def aggregate_across_seeds(df_round: pd.DataFrame) -> pd.DataFrame:
    """
    Average round-level values across seeds so each detector/q/round
    becomes a single plotted point.
    """
    keys = ["detector", "Nm", "q", "p", "attack", "round"]

    agg = (
        df_round.groupby(keys, as_index=False)
                .agg({
                    "y_true": "mean",
                    "score": "mean",
                    "fpr": "mean",
                    "summary_auc": ["mean", "std"],
                    "summary_ttd": ["mean", "std"],
                })
    )

    agg.columns = [
        "detector", "Nm", "q", "p", "attack", "round",
        "y_true_mean", "score_mean", "fpr_mean",
        "summary_auc_mean", "summary_auc_std",
        "summary_ttd_mean", "summary_ttd_std"
    ]
    print(agg)
    return agg


# ============================================================
# Plot 1: summary_auc vs round
# ============================================================
def plot_auc_vs_round(df_seed_avg: pd.DataFrame, attack="backdoor", nm=3, p_value=None):
    plot_df = df_seed_avg[
        (df_seed_avg["attack"] == attack) &
        (df_seed_avg["Nm"] == nm)
    ].copy()

    if p_value is not None:
        plot_df = plot_df[plot_df["p"] == p_value]

    if plot_df.empty:
        print(f"No data for AUC-vs-round plot: attack={attack}, Nm={nm}, p={p_value}")
        return

    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    for (detector, q), g in plot_df.groupby(["detector", "q"]):
        g = g.sort_values("round")
        ax.plot(
            g["round"],
            g["summary_auc_mean"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=f"{detector}, q={q:g}"
        )

        # optional seed variability shading
        y = g["summary_auc_mean"].to_numpy()
        s = g["summary_auc_std"].fillna(0).to_numpy()
        x = g["round"].to_numpy()
        ax.fill_between(x, y - s, y + s, alpha=0.12)

    ax.set_xlabel("Round", fontsize=16)
    ax.set_ylabel("Summary AUC", fontsize=16)
    title = f"Summary AUC vs Round ({attack}, Nm={nm}"
    if p_value is not None:
        title += f", p={p_value:g}"
    title += ")"
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=13, ncol=2)
    style_axes(ax)
    plt.tight_layout()

    suffix = f"{attack}_Nm{nm}"
    if p_value is not None:
        suffix += f"_p{str(p_value).replace('.', '_')}"
    fig.savefig(OUTDIR / f"auc_vs_round_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot 2: mean summary_auc vs q
# ============================================================
def plot_mean_auc_vs_q(df_round: pd.DataFrame, attack="backdoor", nm=3, p_value=None):
    plot_df = df_round[
        (df_round["attack"] == attack) &
        (df_round["Nm"] == nm)
    ].copy()

    if p_value is not None:
        plot_df = plot_df[plot_df["p"] == p_value]

    if plot_df.empty:
        print(f"No data for mean AUC-vs-q plot: attack={attack}, Nm={nm}, p={p_value}")
        return

    # average over seeds and rounds
    agg = (
        plot_df.groupby(["detector", "q"], as_index=False)
               .agg(mean_auc=("summary_auc", "mean"),
                    std_auc=("summary_auc", "std"))
               .sort_values(["detector", "q"])
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.3))

    for detector, g in agg.groupby("detector"):
        g = g.sort_values("q")
        ax.plot(
            g["q"],
            g["mean_auc"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=detector
        )
        ax.fill_between(
            g["q"].to_numpy(),
            (g["mean_auc"] - g["std_auc"].fillna(0)).to_numpy(),
            (g["mean_auc"] + g["std_auc"].fillna(0)).to_numpy(),
            alpha=0.12
        )

    ax.set_xlabel("Availability q")
    ax.set_ylabel("Mean Summary AUC")
    title = f"Mean Summary AUC vs q ({attack}, Nm={nm}"
    if p_value is not None:
        title += f", p={p_value:g}"
    title += ")"
    ax.set_title(title)
    ax.legend(frameon=False)
    style_axes(ax)
    plt.tight_layout()

    suffix = f"{attack}_Nm{nm}"
    if p_value is not None:
        suffix += f"_p{str(p_value).replace('.', '_')}"
    fig.savefig(OUTDIR / f"mean_auc_vs_q_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot 3: TTD vs q
# ============================================================
def plot_ttd_vs_q(df_round: pd.DataFrame, attack="backdoor", nm=3, p_value=None):
    plot_df = df_round[
        (df_round["attack"] == attack) &
        (df_round["Nm"] == nm)
    ].copy()

    if p_value is not None:
        plot_df = plot_df[plot_df["p"] == p_value]

    if plot_df.empty:
        print(f"No data for TTD-vs-q plot: attack={attack}, Nm={nm}, p={p_value}")
        return

    # TTD must be computed per seed first
    ttd_records = []
    keys = ["seed", "detector", "q", "p", "attack", "Nm"]

    for key, g in plot_df.groupby(keys):
        ttd_records.append({
            "seed": key[0],
            "detector": key[1],
            "q": key[2],
            "p": key[3],
            "attack": key[4],
            "Nm": key[5],
            "ttd": first_detection_round(g.sort_values("round"))
        })

    ttd_df = pd.DataFrame(ttd_records).dropna(subset=["ttd"])

    if ttd_df.empty:
        print(f"No detections found for TTD plot: attack={attack}, Nm={nm}, p={p_value}")
        return

    agg = (
        ttd_df.groupby(["detector", "q"], as_index=False)
              .agg(mean_ttd=("ttd", "mean"),
                   std_ttd=("ttd", "std"))
              .sort_values(["detector", "q"])
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.3))

    width = 0.18
    q_vals = sorted(agg["q"].unique())
    detectors = list(agg["detector"].unique())
    x = np.arange(len(q_vals))

    for i, detector in enumerate(detectors):
        g = agg[agg["detector"] == detector].set_index("q").reindex(q_vals)
        means = g["mean_ttd"].to_numpy()
        errs = g["std_ttd"].fillna(0).to_numpy()

        ax.bar(
            x + i * width,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            label=detector,
            alpha=0.9
        )

    ax.set_xticks(x + width * (len(detectors) - 1) / 2)
    ax.set_xticklabels([f"{q:g}" for q in q_vals])
    ax.set_xlabel("Availability q")
    ax.set_ylabel("TTD (first detected round)")
    title = f"TTD vs q ({attack}, Nm={nm}"
    if p_value is not None:
        title += f", p={p_value:g}"
    title += ")"
    ax.set_title(title)
    ax.legend(frameon=False)
    style_axes(ax)
    plt.tight_layout()

    suffix = f"{attack}_Nm{nm}"
    if p_value is not None:
        suffix += f"_p{str(p_value).replace('.', '_')}"
    fig.savefig(OUTDIR / f"ttd_vs_q_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot 4: score distribution by detector
# ============================================================
def plot_score_boxplot(df_round: pd.DataFrame, attack="backdoor", nm=3, p_value=None):
    """
    Use round-collapsed data, not raw data, so Nm=3 does not triple-count each round.
    """
    plot_df = df_round[
        (df_round["attack"] == attack) &
        (df_round["Nm"] == nm)
    ].copy()

    if p_value is not None:
        plot_df = plot_df[plot_df["p"] == p_value]

    if plot_df.empty:
        print(f"No data for score boxplot: attack={attack}, Nm={nm}, p={p_value}")
        return

    detectors = sorted(plot_df["detector"].unique())
    data = [
        plot_df.loc[plot_df["detector"] == d, "score"].dropna().values
        for d in detectors
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=detectors, showfliers=False)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Round-level score")
    ax.set_title(f"Score Distribution by Detector ({attack}, Nm={nm}, p={p_value:g})")

    # helpful because ROBUST can have very large scores
    if plot_df["score"].max() / max(plot_df["score"].median(), 1e-9) > 50:
        ax.set_yscale("symlog", linthresh=1.0)

    style_axes(ax)
    plt.tight_layout()

    suffix = f"{attack}_Nm{nm}"
    if p_value is not None:
        suffix += f"_p{str(p_value).replace('.', '_')}"
    fig.savefig(OUTDIR / f"score_boxplot_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)






    def aggregate_round_level1(df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        group_cols = ["seed", "detector", "Nm", "q", "p", "attack", "round"]

        for keys, sub in df.groupby(group_cols):
            seed, detector, nm, q, p, attack, rnd = keys

            rows.append({
                "seed": seed,
                "detector": detector,
                "Nm": nm,
                "q": q,
                "p": p,
                "attack": attack,
                "round": rnd,
                "y_true": sub["y_true"].max(),
                "score": sub["score"].mean(),
                #"flags": any(has_flag_value(v) for v in sub["flags"]),
                "fpr": sub["fpr"].mean() if "fpr" in sub.columns else np.nan,
                "summary_auc": sub["summary_auc"].mean() if "summary_auc" in sub.columns else np.nan,
                "summary_ttd": sub["summary_ttd"].mean() if "summary_ttd" in sub.columns else np.nan,
            })

        return pd.DataFrame(rows)

    def aggregate_across_seeds1(df_round: pd.DataFrame) -> pd.DataFrame:
        agg = (
            df_round
            .groupby(["detector", "Nm", "q", "p", "attack", "round"], as_index=False)
            .agg(
                auc_mean=("summary_auc", "mean"),
                auc_std=("summary_auc", "std"),
                ttd_mean=("summary_ttd", "mean"),
                ttd_std=("summary_ttd", "std"),
                score_mean=("score", "mean"),
                score_std=("score", "std"),
                n=("seed", "nunique"),
            )
        )

        return agg

    def plot_auc_vs_round_by_attack(df_seed_avg: pd.DataFrame, detector="MMR", nm=3, p_value=None, q_value=None):
        plot_df = df_seed_avg[df_seed_avg["Nm"] == nm].copy()

        if detector is not None:
            plot_df = plot_df[plot_df["detector"] == detector]

        if p_value is not None:
            plot_df = plot_df[plot_df["p"] == p_value]

        if q_value is not None:
            plot_df = plot_df[plot_df["q"] == q_value]

        if plot_df.empty:
            print(f"No data for AUC-vs-round-by-attack: detector={detector}, Nm={nm}, p={p_value}, q={q_value}")
            return

        fig, ax = plt.subplots(figsize=(7, 4.5))

        attack_order = ["backdoor", "scaled-gradient", "slow-drift"]
        attack_styles = {
            "backdoor": {"linestyle": "-", "marker": "o"},
            "coordinated-scaling": {"linestyle": "--", "marker": "s"},
            "slow-drift": {"linestyle": ":", "marker": "^"},
        }

        for attack in attack_order:
            sub = plot_df[plot_df["attack"] == attack].sort_values("round")
            if sub.empty:
                print(f"Skipping {attack}: no rows")
                continue

            x = sub["round"].to_numpy()
            y = sub["auc_mean"].to_numpy(dtype=float)
            sd = sub["auc_std"].fillna(0.0).to_numpy(dtype=float)

            style = attack_styles.get(attack, {})
            ax.plot(x, y, linewidth=2.6, markersize=6, label=attack, **style)
            ax.fill_between(x, y - sd, y + sd, alpha=0.12)

        ax.set_xlabel("Round")
        ax.set_ylabel("AUC")
        title = f"AUC vs Round by Attack ({detector}, Nm={nm}"
        if p_value is not None:
            title += f", p={p_value:g}"
        if q_value is not None:
            title += f", q={q_value:g}"
        title += ")"
        ax.set_title(title)

        ax.set_ylim(0, 1.05)
        style_axes(ax)
        ax.legend(frameon=False)

        plt.tight_layout()

        suffix = f"{detector}_Nm{nm}"
        if p_value is not None:
            suffix += f"_p{str(p_value).replace('.', '_')}"
        if q_value is not None:
            suffix += f"_q{str(q_value).replace('.', '_')}"

        fig.savefig(OUTDIR / f"auc_vs_round_by_attack_{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ============================================================
    # Main
    # ============================================================
    if __name__ == "__main__":
        df = load_results(INPUT_FILE)
        df_round = aggregate_round_level1(df)
        df_seed_avg = aggregate_across_seeds1(df_round)

        print("Loaded shape:", df.shape)
        print("Round-level shape:", df_round.shape)
        print("Seed-averaged shape:", df_seed_avg.shape)
        print("Detectors:", sorted(df["detector"].dropna().unique()))
        print("q values:", sorted(df["q"].dropna().unique()))
        print("p values:", sorted(df["p"].dropna().unique()))
        print("Nm values:", sorted(df["Nm"].dropna().unique()))
        print("Attack values:", sorted(df["attack"].dropna().unique()))

        NM = 3
        p_values = sorted(df["p"].dropna().unique())
        q_values = sorted(df["q"].dropna().unique())

        for p_value in p_values:
            for q_value in q_values:
                plot_auc_vs_round_by_attack(
                    df_seed_avg,
                    detector="MMR",
                    nm=NM,
                    p_value=p_value,
                    q_value=q_value
                )

        print(f"Plots saved in: {OUTDIR.resolve()}")

