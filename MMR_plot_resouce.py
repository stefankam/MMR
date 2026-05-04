import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# -----------------------------
# Input MMR data
# -----------------------------
rounds = "1	 2	3	4	5	6	7	8	9	10"
cpu_voluntary_ctxt_switches	= "3474	3999	4391	4998	5744	6134	6523	6921	7290	7865"
cpu_nonvoluntary_ctxt_switches	= "109	156	198	239	273	300	351	391	429	469"
mem_vm_size= 	"11483036	17142172	17414568	17594280	17594280	17967260	18096284	18057884	18092444	18057116"
mem_vm_rss=	"2116252	5892852	6167236	6347360	6347540	6720388	6849444	6811184	6845644	6810548"
mem_vm_hwm=	"2670836	6827424	8561212	8711256	8845772	8845880	9218868	9347788	9347788	9347788"
mem_vm_swap=	"0	0	0	0	0	0	0	0	0	0"
disk_rss_file=	"677132	513424	514620	514620	514620	514620	514620	514620	514620	514620"
disk_rss_shmem=	"4	4	4	4	4	4	4	4	4	4"
overhead_round_total_sec=	"32.47180472	24.32109131	21.53043169	20.84883772	20.8764341	21.6338885	21.23630314	21.2977828	21.68777118	21.29442673"
overhead_detection_sec=	"0.578806054	0.57514317	0.513256275	0.530693763	0.494372159	0.523468417	0.534232359	0.586426343	0.556242336	0.512424762"


# Convert to numpy arrays
data = {
    "step": np.array(rounds.split(), dtype=float),
    "cpu_voluntary_ctxt_switches": np.array(cpu_voluntary_ctxt_switches.split(), dtype=float),
    "cpu_nonvoluntary_ctxt_switches": np.array(cpu_nonvoluntary_ctxt_switches.split(), dtype=float),
    "mem_vm_size": np.array(mem_vm_size.split(), dtype=float),
    "mem_vm_rss": np.array(mem_vm_rss.split(), dtype=float),
    "mem_vm_hwm": np.array(mem_vm_hwm.split(), dtype=float),
    "mem_vm_swap": np.array(mem_vm_swap.split(), dtype=float),
    "disk_rss_file": np.array(disk_rss_file.split(), dtype=float),
    "disk_rss_shmem": np.array(disk_rss_shmem.split(), dtype=float),
    "overhead_round_total_sec": np.array(overhead_round_total_sec.split(), dtype=float),
    "overhead_detection_sec": np.array(overhead_detection_sec.split(), dtype=float),
}


mmr_df = pd.DataFrame(data)



# -----------------------------
# Input ROBUST data
# -----------------------------
rounds = "1	 2	3	4	5	6	7	8	9	10"
cpu_voluntary_ctxt_switches=	"318010	318461	319340	320036	320568	321205	321804	322389	323014	323879"
cpu_nonvoluntary_ctxt_switches=	"18799	18829	18872	18912	18944	18987	19013	19055	19101	19155"
mem_vm_size=	"15635984	19522176	19522176	19672952	19672952	19823728	20125280	19974504	19974504	19974504"
mem_vm_rss=	"4151092	8047600	8047744	8198508	8198596	8349244	8650720	8500172	8499964	8500120"
mem_vm_hwm=	"13078788	13078788	13078788	13078788	13078788	13078788	13078788	13078788	13078788	13078788"
mem_vm_swap=	"0	0	0	0	0	0	0	0	0	0"
disk_rss_file=	"673496	515380	514832	514832	514832	514832	514832	514832	514832	514832"
disk_rss_shmem=	"4	4	4	4	4	4	4	4	4	4"
overhead_round_total_sec=	"23.0792255	21.37334832	20.96381785	21.05577619	21.27458231	21.03479179	21.16739173	21.21321632	20.96915157	21.12961386"
overhead_detection_sec=	"1.433972087	1.399014566	1.394310871	1.410435838	1.394154636	1.404236642	1.42923903	1.401493365	1.394368304	1.397672198"


# Convert to numpy arrays
data = {
    "step": np.array(rounds.split(), dtype=float),
    "cpu_voluntary_ctxt_switches": np.array(cpu_voluntary_ctxt_switches.split(), dtype=float),
    "cpu_nonvoluntary_ctxt_switches": np.array(cpu_nonvoluntary_ctxt_switches.split(), dtype=float),
    "mem_vm_size": np.array(mem_vm_size.split(), dtype=float),
    "mem_vm_rss": np.array(mem_vm_rss.split(), dtype=float),
    "mem_vm_hwm": np.array(mem_vm_hwm.split(), dtype=float),
    "mem_vm_swap": np.array(mem_vm_swap.split(), dtype=float),
    "disk_rss_file": np.array(disk_rss_file.split(), dtype=float),
    "disk_rss_shmem": np.array(disk_rss_shmem.split(), dtype=float),
    "overhead_round_total_sec": np.array(overhead_round_total_sec.split(), dtype=float),
    "overhead_detection_sec": np.array(overhead_detection_sec.split(), dtype=float),
}


robust_df = pd.DataFrame(data)


# -----------------------------
# Input None data
# -----------------------------

rounds = "1	 2	3	4	5	6	7	8	9	10"
cpu_voluntary_ctxt_switches=	"641989	642384	642988	643375	643762	644153	644796	645184	645850	646243"
cpu_nonvoluntary_ctxt_switches=	"43181	43226	43256	43283	43313	43347	43379	43409	43434	43459"
mem_vm_size=	"16274544	19086880	19086880	19237656	19237656	19539208	19388432	19689984	19388432	19539208"
mem_vm_rss=	"4768584	7614716	7614508	7765272	7765296	8066900	7916040	8217628	7916164	8066676"
mem_vm_hwm=	"14181256	14181256	14181256	14181256	14181256	14181256	14181256	14181256	14181256	14181256"
mem_vm_swap=	"0	0	0	0	0	0	0	0	0	0"
disk_rss_file=	"650268	516908	515012	514840	514840	514840	514840	514840	514840	514840"
disk_rss_shmem=	"4	4	4	4	4	4	4	4	4	4"
overhead_round_total_sec=	"21.73847139	19.84804572	19.43462203	19.76708052	19.89348947	19.72858301	19.99367794	20.08880144	20.29085189	19.88904935"
overhead_detection_sec=	"3.41E-06	3.51E-06	3.48E-06	3.50E-06	4.49E-06	3.81E-06	3.72E-06	3.10E-06	2.89E-06	5.17E-06"


# Convert to numpy arrays
data = {
    "step": np.array(rounds.split(), dtype=float),
    "cpu_voluntary_ctxt_switches": np.array(cpu_voluntary_ctxt_switches.split(), dtype=float),
    "cpu_nonvoluntary_ctxt_switches": np.array(cpu_nonvoluntary_ctxt_switches.split(), dtype=float),
    "mem_vm_size": np.array(mem_vm_size.split(), dtype=float),
    "mem_vm_rss": np.array(mem_vm_rss.split(), dtype=float),
    "mem_vm_hwm": np.array(mem_vm_hwm.split(), dtype=float),
    "mem_vm_swap": np.array(mem_vm_swap.split(), dtype=float),
    "disk_rss_file": np.array(disk_rss_file.split(), dtype=float),
    "disk_rss_shmem": np.array(disk_rss_shmem.split(), dtype=float),
    "overhead_round_total_sec": np.array(overhead_round_total_sec.split(), dtype=float),
    "overhead_detection_sec": np.array(overhead_detection_sec.split(), dtype=float),
}

none_df = pd.DataFrame(data)


# -----------------------------
# Input Flanders data
# -----------------------------

rounds = "1	 2	3	4	5	6	7	8	9	10"
cpu_voluntary_ctxt_switches=	"946756	947162	947559	947953	948380	948763	949410	950062	950638	951353"
cpu_nonvoluntary_ctxt_switches=	"62977	63020	63053	63112	63166	63228	63292	63363	63421	63505"
mem_vm_size=	"15910460	19325900	19660220	19961772	20281756	20866428	21341548	21832800	22330960	22648396"
mem_vm_rss=	"4411072	7853712	8156356	8458200	8778316	9362992	9837784	10329160	10826892	11144576"
mem_vm_hwm=	"14181256	14181256	14181256	14181256	14181256	16155904	18057940	18544796	18557316	19034128"
mem_vm_swap=	"0	0	0	0	0	0	0	0	0	0"
disk_rss_file=	"656808	516616	516176	516296	516360	516360	516360	516360	516360	516360"
disk_rss_shmem=	"4	4	4	4	4	4	4	4	4	4"
overhead_round_total_sec=	"22.34511124	22.3433895	23.45793042	24.52445485	25.42335349	27.60075728	26.27122716	25.98302364	57.82795674	27.19741088"
overhead_detection_sec=	"3.13E-05	1.53513637	2.460160175	3.41119773	4.607561715	6.963311893	5.675997446	5.078346739	7.731698795	5.712372693"


# Convert to numpy arrays
data = {
    "step": np.array(rounds.split(), dtype=float),
    "cpu_voluntary_ctxt_switches": np.array(cpu_voluntary_ctxt_switches.split(), dtype=float),
    "cpu_nonvoluntary_ctxt_switches": np.array(cpu_nonvoluntary_ctxt_switches.split(), dtype=float),
    "mem_vm_size": np.array(mem_vm_size.split(), dtype=float),
    "mem_vm_rss": np.array(mem_vm_rss.split(), dtype=float),
    "mem_vm_hwm": np.array(mem_vm_hwm.split(), dtype=float),
    "mem_vm_swap": np.array(mem_vm_swap.split(), dtype=float),
    "disk_rss_file": np.array(disk_rss_file.split(), dtype=float),
    "disk_rss_shmem": np.array(disk_rss_shmem.split(), dtype=float),
    "overhead_round_total_sec": np.array(overhead_round_total_sec.split(), dtype=float),
    "overhead_detection_sec": np.array(overhead_detection_sec.split(), dtype=float),
}

flanders_df = pd.DataFrame(data)

# -----------------------------
# Output directory
# -----------------------------
outdir = Path("neurips_resource_figures")
outdir.mkdir(parents=True, exist_ok=True)



# -----------------------------
# Global plotting config
# -----------------------------
plt.rcParams.update({
    # Figure
    "figure.figsize": (7.2, 4.6),   # slightly larger → better readability
    "figure.dpi": 200,
    "savefig.dpi": 300,

    # Fonts
    "font.family": "serif",
    "font.size": 13,                # base font (↑ from 11)
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    # Lines & axes
    "axes.linewidth": 1.2,          # stronger axes
    "lines.linewidth": 2.6,         # more visible curves
    "lines.markersize": 6,

    # Grid
    "grid.linewidth": 0.8,
    "grid.alpha": 0.3,

    # Export (VERY important for papers)
    "pdf.fonttype": 42,             # editable text
    "ps.fonttype": 42,
})

def kb_to_gb(x):
    return x / (1024 * 1024)

def style_axes(ax):
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def add_method_and_metric_legends(ax, methods, colors, metric_styles, method_loc="upper left", metric_loc="upper right"):
    method_handles = [
        Line2D([0], [0], color=colors[name], lw=2.2, linestyle="-")
        for name in methods.keys()
    ]
    method_labels = list(methods.keys())

    legend1 = ax.legend(
        method_handles,
        method_labels,
        title="Method",
        loc=method_loc,
        frameon=False
    )
    ax.add_artist(legend1)

    metric_handles = [
        Line2D([0], [0], color="black", lw=2.2, linestyle=style)
        for _, style in metric_styles
    ]
    metric_labels = [label for label, _ in metric_styles]

    ax.legend(
        metric_handles,
        metric_labels,
        title="Metric",
        loc=metric_loc,
        frameon=False
    )

methods = {
    "Flanders": flanders_df,
    "MMR": mmr_df,
    "Robust": robust_df,
    "None": none_df,
}

colors = {
    "Flanders": "#1f77b4",
    "MMR": "#d62728",
    "Robust": "#2ca02c",
    "None": "#7f7f7f",
}

# -----------------------------
# 1) Memory comparison
# -----------------------------
fig, ax = plt.subplots()

for name, df in methods.items():
    ax.plot(
        df["step"], kb_to_gb(df["mem_vm_rss"]),
        color=colors[name], linestyle="-"
    )
    ax.plot(
        df["step"], kb_to_gb(df["mem_vm_size"]),
        color=colors[name], linestyle="--", alpha=0.85
    )

ax.set_xlabel("Step / Round")
ax.set_ylabel("Memory (GB)")
ax.set_title("Memory Usage Comparison Across Methods")
style_axes(ax)
add_method_and_metric_legends(
    ax, methods, colors,
    metric_styles=[("RSS", "-"), ("VM Size", "--")]
)

plt.tight_layout()
fig.savefig(outdir / "memory_comparison.pdf", bbox_inches="tight")
fig.savefig(outdir / "memory_comparison.png", bbox_inches="tight")
plt.close(fig)

# -----------------------------
# 2) Context switches comparison
# -----------------------------
fig, ax = plt.subplots()

for name, df in methods.items():
    ax.plot(
        df["step"], df["cpu_voluntary_ctxt_switches"],
        color=colors[name], linestyle="-"
    )
    ax.plot(
        df["step"], df["cpu_nonvoluntary_ctxt_switches"],
        color=colors[name], linestyle=":"
    )

ax.set_xlabel("Step / Round")
ax.set_ylabel("Context Switch Count")
ax.set_title("CPU Context Switches Comparison")
style_axes(ax)
add_method_and_metric_legends(
    ax, methods, colors,
    metric_styles=[("Voluntary", "-"), ("Non-voluntary", ":")]
)

plt.tight_layout()
fig.savefig(outdir / "context_switches_comparison.pdf", bbox_inches="tight")
fig.savefig(outdir / "context_switches_comparison.png", bbox_inches="tight")
plt.close(fig)

# -----------------------------
# 3) Overhead comparison
# -----------------------------
fig, ax = plt.subplots()

for name, df in methods.items():
    ax.plot(
        df["step"], df["overhead_round_total_sec"],
        color=colors[name], linestyle="-"
    )
    ax.plot(
        df["step"], df["overhead_detection_sec"],
        color=colors[name], linestyle="--"
    )

ax.set_xlabel("Step / Round")
ax.set_ylabel("Time (s)")
ax.set_title("System Overhead Comparison Across Methods")
style_axes(ax)
add_method_and_metric_legends(
    ax, methods, colors,
    metric_styles=[("Total Round Overhead", "-"), ("Detection Overhead", "--")]
)

plt.tight_layout()
fig.savefig(outdir / "overhead_comparison.pdf", bbox_inches="tight")
fig.savefig(outdir / "overhead_comparison.png", bbox_inches="tight")
plt.close(fig)

# -----------------------------
# 4) Detection fraction comparison
# -----------------------------
fig, ax = plt.subplots()

for name, df in methods.items():
    fraction = df["overhead_detection_sec"] / df["overhead_round_total_sec"]
    ax.plot(
        df["step"], fraction,
        label=name,
        color=colors[name], linestyle="-"
    )

ax.set_xlabel("Step / Round")
ax.set_ylabel("Detection / Total Overhead")
ax.set_title("Detection Cost Ratio Across Methods")
style_axes(ax)
ax.legend(frameon=False, title="Method")

plt.tight_layout()
fig.savefig(outdir / "detection_fraction_comparison.pdf", bbox_inches="tight")
fig.savefig(outdir / "detection_fraction_comparison.png", bbox_inches="tight")
plt.close(fig)

# -----------------------------
# 5) Disk-backed memory comparison
# -----------------------------
fig, ax = plt.subplots()

for name, df in methods.items():
    ax.plot(
        df["step"], kb_to_gb(df["disk_rss_file"]),
        color=colors[name], linestyle="-"
    )
    ax.plot(
        df["step"], kb_to_gb(df["disk_rss_shmem"]),
        color=colors[name], linestyle="--"
    )

ax.set_xlabel("Step / Round")
ax.set_ylabel("Memory (GB)")
ax.set_title("Disk-backed Memory Comparison")
style_axes(ax)
add_method_and_metric_legends(
    ax, methods, colors,
    metric_styles=[("File-backed RSS", "-"), ("Shared Memory RSS", "--")]
)

plt.tight_layout()
fig.savefig(outdir / "disk_memory_comparison.pdf", bbox_inches="tight")
fig.savefig(outdir / "disk_memory_comparison.png", bbox_inches="tight")
plt.close(fig)

print(f"Saved figures to: {outdir.resolve()}")


















import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import sem

# -----------------------------
# Configuration
# -----------------------------
INPUT_FILE = "resource.csv"
OUTDIR = Path("neurips_resource_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT_FILE)
df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

# Fix column names (your CSV uses detector, not method)
df["detector"] = df["detector"].astype(str).str.strip()

# Create step (round index per seed + detector + q)
df["step"] = df.groupby(["seed", "detector", "q"]).cumcount() + 1

# Convert numeric
numeric_cols = [
    "cpu_voluntary_ctxt_switches",
    "cpu_nonvoluntary_ctxt_switches",
    "mem_vm_size",
    "mem_vm_rss",
    "mem_vm_hwm",
    "mem_vm_swap",
    "disk_rss_file",
    "disk_rss_shmem",
    "overhead_round_total_sec",
    "overhead_detection_sec",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Aggregate across seeds (mean + CI)
# -----------------------------
def aggregate_with_ci(df, metric):
    grouped = df.groupby(["detector", "q", "step"])[metric]

    mean = grouped.mean()
    ci = grouped.apply(lambda x: 1.96 * sem(x) if len(x) > 1 else 0)

    out = pd.DataFrame({
        "mean": mean,
        "ci": ci
    }).reset_index()

    return out

# Example: memory RSS
mem_rss_agg = aggregate_with_ci(df, "mem_vm_rss")
mem_size_agg = aggregate_with_ci(df, "mem_vm_size")

overhead_total_agg = aggregate_with_ci(df, "overhead_round_total_sec")
overhead_det_agg = aggregate_with_ci(df, "overhead_detection_sec")

# -----------------------------
# Plot settings
# -----------------------------
plt.rcParams.update({
    "figure.figsize": (6.2, 4.0),
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 11,
})

colors = {
    "FLANDERS": "#1f77b4",
    "MMR": "#ff7f0e",
    "ROBUST": "#d62728",
    "NONE": "#7f7f7f",
}

def kb_to_gb(x):
    return x / (1024 * 1024)

def plot_with_ci(ax, data, label, color, linestyle="-", convert=None):
    x = data["step"]
    y = data["mean"]
    ci = data["ci"]

    if convert:
        y = convert(y)
        ci = convert(ci)

    ax.plot(x, y, label=label, color=color, linestyle=linestyle)
    ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2)

def style_axes(ax):
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# -----------------------------
# Filter (optional): choose q
# -----------------------------
q_value = 0.2   # change to 0.5 / 1.0 if needed

# -----------------------------
# 1) Memory plot
# -----------------------------
fig, ax = plt.subplots()

for method in ["MMR", "FLANDERS", "ROBUST", "NONE"]:
    d1 = mem_rss_agg[(mem_rss_agg["detector"] == method) & (mem_rss_agg["q"] == q_value)]
    d2 = mem_size_agg[(mem_size_agg["detector"] == method) & (mem_size_agg["q"] == q_value)]

    if len(d1) == 0:
        continue

    plot_with_ci(ax, d1, f"{method} RSS", colors[method], "-", kb_to_gb)
    plot_with_ci(ax, d2, f"{method} VM", colors[method], "--", kb_to_gb)

ax.set_xlabel("Round")
ax.set_ylabel("Memory (GB)")
ax.set_title(f"Memory Usage (q={q_value})")
style_axes(ax)
ax.legend(frameon=False, ncol=2)

plt.tight_layout()
fig.savefig(OUTDIR / f"memory_q{q_value}.pdf", bbox_inches="tight")
plt.close(fig)

# -----------------------------
# 2) Overhead plot
# -----------------------------
fig, ax = plt.subplots()

for method in ["MMR", "FLANDERS", "ROBUST", "NONE"]:
    d1 = overhead_total_agg[(overhead_total_agg["detector"] == method) & (overhead_total_agg["q"] == q_value)]
    d2 = overhead_det_agg[(overhead_det_agg["detector"] == method) & (overhead_det_agg["q"] == q_value)]

    if len(d1) == 0:
        continue

    plot_with_ci(ax, d1, f"{method} total", colors[method], "-")
    plot_with_ci(ax, d2, f"{method} detect", colors[method], "--")

ax.set_xlabel("Round")
ax.set_ylabel("Time (s)")
ax.set_title(f"Overhead (q={q_value})")
style_axes(ax)
ax.legend(frameon=False)

plt.tight_layout()
fig.savefig(OUTDIR / f"overhead_q{q_value}.pdf", bbox_inches="tight")
plt.close(fig)

print(f"Saved figures to: {OUTDIR.resolve()}")







import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
from scipy.stats import sem

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = "resource.csv"
OUTDIR = Path("neurips_resource_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Which q values to plot; set to None to use all values found in the CSV
Q_VALUES = None

# Methods in preferred display order
METHOD_ORDER = ["MMR", "FLANDERS", "ROBUST", "NONE"]

# Colors by method
COLORS = {
    "MMR": "#ff7f0e",
    "FLANDERS": "#1f77b4",
    "ROBUST": "#d62728",
    "NONE": "#7f7f7f",
}

# ============================================================
# Global plotting config
# ============================================================
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
# Helpers
# ============================================================

def style_axes(ax):
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=14,
        width=1.3,
        length=5
    )

def kb_to_gb(x):
    return np.asarray(x) / (1024.0 * 1024.0)

def add_method_and_metric_legends(ax, methods_present, metric_styles,
                                  method_loc="upper left", metric_loc="upper right"):
    method_handles = [
        Line2D([0], [0], color=COLORS[m], lw=2.8, linestyle="-")
        for m in methods_present
    ]
    legend1 = ax.legend(
        method_handles,
        methods_present,
        title="Method",
        loc=method_loc,
        frameon=False
    )
    ax.add_artist(legend1)

    metric_handles = [
        Line2D([0], [0], color="black", lw=2.8, linestyle=style)
        for _, style in metric_styles
    ]
    metric_labels = [label for label, _ in metric_styles]

    ax.legend(
        metric_handles,
        metric_labels,
        title="Metric",
        loc=metric_loc,
        frameon=False
    )

def plot_mean_ci(ax, subdf, x_col, mean_col, ci_col, color, label=None, linestyle="-", transform=None):
    x = subdf[x_col].to_numpy()
    y = subdf[mean_col].to_numpy()
    ci = subdf[ci_col].to_numpy()

    if transform is not None:
        y = transform(y)
        ci = transform(ci)

    ax.plot(x, y, color=color, linestyle=linestyle, label=label)
    ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.22)

def safe_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(INPUT_FILE)
df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

required_cols = [
    "seed", "detector", "q",
    "cpu_voluntary_ctxt_switches",
    "cpu_nonvoluntary_ctxt_switches",
    "mem_vm_size", "mem_vm_rss", "mem_vm_hwm", "mem_vm_swap",
    "disk_rss_file", "disk_rss_shmem",
    "overhead_round_total_sec", "overhead_detection_sec"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in {INPUT_FILE}: {missing}")

df["detector"] = df["detector"].astype(str).str.strip().str.upper()

numeric_cols = [
    "seed", "q",
    "cpu_voluntary_ctxt_switches",
    "cpu_nonvoluntary_ctxt_switches",
    "mem_vm_size", "mem_vm_rss", "mem_vm_hwm", "mem_vm_swap",
    "disk_rss_file", "disk_rss_shmem",
    "overhead_round_total_sec", "overhead_detection_sec"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Create round/step index within each seed-detector-q trajectory
df = df.sort_values(["seed", "detector", "q"]).copy()
df["step"] = df.groupby(["seed", "detector", "q"]).cumcount() + 1

# Detection ratio per row
df["detection_fraction"] = safe_ratio(df["overhead_detection_sec"], df["overhead_round_total_sec"])

# q values to use
if Q_VALUES is None:
    q_values = sorted(df["q"].dropna().unique().tolist())
else:
    q_values = Q_VALUES

# Methods actually present
methods_present = [m for m in METHOD_ORDER if m in set(df["detector"].unique())]

# ============================================================
# Aggregate mean + 95% CI over seeds
# ============================================================
metrics = [
    "cpu_voluntary_ctxt_switches",
    "cpu_nonvoluntary_ctxt_switches",
    "mem_vm_size",
    "mem_vm_rss",
    "mem_vm_hwm",
    "mem_vm_swap",
    "disk_rss_file",
    "disk_rss_shmem",
    "overhead_round_total_sec",
    "overhead_detection_sec",
    "detection_fraction",
]

agg_frames = []
for metric in metrics:
    grouped = df.groupby(["detector", "q", "step"])[metric]
    summary = grouped.agg(["mean", "count", "std"]).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    summary["ci95"] = 1.96 * summary["sem"].fillna(0.0)
    summary["metric"] = metric
    agg_frames.append(summary)

agg = pd.concat(agg_frames, ignore_index=True)

# ============================================================
# Plotting routines
# ============================================================
def savefig(fig, stem):
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)

def plot_memory_for_q(q_val):
    fig, ax = plt.subplots()

    for method in methods_present:
        rss = agg[
            (agg["metric"] == "mem_vm_rss") &
            (agg["detector"] == method) &
            (agg["q"] == q_val)
        ].sort_values("step")
        vms = agg[
            (agg["metric"] == "mem_vm_size") &
            (agg["detector"] == method) &
            (agg["q"] == q_val)
        ].sort_values("step")

        if not rss.empty:
            plot_mean_ci(ax, rss, "step", "mean", "ci95", COLORS[method], linestyle="-", transform=kb_to_gb)
        if not vms.empty:
            plot_mean_ci(ax, vms, "step", "mean", "ci95", COLORS[method], linestyle="--", transform=kb_to_gb)

    ax.set_xlabel("Round")
    ax.set_ylabel("Memory (GB)")
    ax.set_title(f"Memory Usage Comparison (q={q_val})")
    style_axes(ax)
    add_method_and_metric_legends(
        ax, methods_present,
        metric_styles=[("RSS", "-"), ("VM Size", "--")]
    )
    savefig(fig, f"memory_comparison_q{q_val}")

def plot_cpu_for_q(q_val):
    fig, ax = plt.subplots()

    for method in methods_present:
        voluntary = agg[
            (agg["metric"] == "cpu_voluntary_ctxt_switches") &
            (agg["detector"] == method) &
            (agg["q"] == q_val)
        ].sort_values("step")
        nonvol = agg[
            (agg["metric"] == "cpu_nonvoluntary_ctxt_switches") &
            (agg["detector"] == method) &
            (agg["q"] == q_val)
        ].sort_values("step")

        if not voluntary.empty:
            plot_mean_ci(ax, voluntary, "step", "mean", "ci95", COLORS[method], linestyle="-")
        if not nonvol.empty:
            plot_mean_ci(ax, nonvol, "step", "mean", "ci95", COLORS[method], linestyle=":")

    ax.set_xlabel("Round")
    ax.set_ylabel("Context Switch Count")
    ax.set_title(f"CPU Context Switches Comparison (q={q_val})")
    style_axes(ax)
    add_method_and_metric_legends(
        ax, methods_present,
        metric_styles=[("Voluntary", "-"), ("Non-voluntary", ":")]
    )
    savefig(fig, f"cpu_context_switches_q{q_val}")

def plot_disk_for_q(q_val):
    fig, ax = plt.subplots()

    for method in methods_present:
        file_backed = agg[
            (agg["metric"] == "disk_rss_file") &
            (agg["detector"] == method) &
            (agg["q"] == q_val)
        ].sort_values("step")
        shmem = agg[
            (agg["metric"] == "disk_rss_shmem") &
            (agg["detector"] == method) &
            (agg["q"] == q_val)
        ].sort_values("step")

        if not file_backed.empty:
            plot_mean_ci(ax, file_backed, "step", "mean", "ci95", COLORS[method], linestyle="-", transform=kb_to_gb)
        if not shmem.empty:
            plot_mean_ci(ax, shmem, "step", "mean", "ci95", COLORS[method], linestyle="--", transform=kb_to_gb)

    ax.set_xlabel("Round")
    ax.set_ylabel("Disk-backed Memory (GB)")
    ax.set_title(f"Disk-backed Memory Comparison (q={q_val})")
    style_axes(ax)
    add_method_and_metric_legends(
        ax, methods_present,
        metric_styles=[("File-backed RSS", "-"), ("Shared Memory RSS", "--")]
    )
    savefig(fig, f"disk_memory_comparison_q{q_val}")

def plot_overhead_for_q(q_val):
    fig, ax = plt.subplots()

    for method in methods_present:
        total = agg[
            (agg["metric"] == "overhead_round_total_sec") &
            (agg["detector"] == method) &
            (agg["q"] == q_val)
        ].sort_values("step")
        det = agg[
            (agg["metric"] == "overhead_detection_sec") &
            (agg["detector"] == method) &
            (agg["q"] == q_val)
        ].sort_values("step")

        if not total.empty:
            plot_mean_ci(ax, total, "step", "mean", "ci95", COLORS[method], linestyle="-")
        if not det.empty:
            plot_mean_ci(ax, det, "step", "mean", "ci95", COLORS[method], linestyle="--")

    ax.set_xlabel("Round")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"System Overhead Comparison (q={q_val})")
    style_axes(ax)
    add_method_and_metric_legends(
        ax, methods_present,
        metric_styles=[("Total Round Overhead", "-"), ("Detection Overhead", "--")]
    )
    savefig(fig, f"overhead_comparison_q{q_val}")

def plot_detection_fraction_for_q(q_val):
    fig, ax = plt.subplots()

    for method in methods_present:
        frac = agg[
            (agg["metric"] == "detection_fraction") &
            (agg["detector"] == method) &
            (agg["q"] == q_val)
        ].sort_values("step")

        if not frac.empty:
            plot_mean_ci(ax, frac, "step", "mean", "ci95", COLORS[method], label=method, linestyle="-")

    ax.set_xlabel("Round")
    ax.set_ylabel("Detection / Total Overhead")
    ax.set_title(f"Detection Cost Ratio (q={q_val})")
    style_axes(ax)
    ax.legend(frameon=False, title="Method")
    savefig(fig, f"detection_fraction_comparison_q{q_val}")

# ============================================================
# Summary plots across q
# ============================================================
def summarize_final_step(metric):
    rows = []
    for method in methods_present:
        for q_val in q_values:
            sub = agg[
                (agg["metric"] == metric) &
                (agg["detector"] == method) &
                (agg["q"] == q_val)
            ].sort_values("step")
            if sub.empty:
                continue
            last = sub.iloc[-1]
            rows.append({
                "detector": method,
                "q": q_val,
                "mean": last["mean"],
                "ci95": last["ci95"],
            })
    return pd.DataFrame(rows)

def plot_metric_vs_q(metric, ylabel, title, stem, transform=None):
    summary = summarize_final_step(metric)
    fig, ax = plt.subplots()

    for method in methods_present:
        sub = summary[summary["detector"] == method].sort_values("q")
        if sub.empty:
            continue

        x = sub["q"].to_numpy()
        y = sub["mean"].to_numpy()
        ci = sub["ci95"].to_numpy()

        if transform is not None:
            y = transform(y)
            ci = transform(ci)

        ax.plot(x, y, marker="o", color=COLORS[method], label=method)
        ax.fill_between(x, y - ci, y + ci, color=COLORS[method], alpha=0.22)

    ax.set_xlabel("Availability q")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    style_axes(ax)
    ax.legend(frameon=False, title="Method")
    savefig(fig, stem)

# ============================================================
# Generate all per-q plots
# ============================================================
for q_val in q_values:
    plot_memory_for_q(q_val)
    plot_cpu_for_q(q_val)
    plot_disk_for_q(q_val)
    plot_overhead_for_q(q_val)
    plot_detection_fraction_for_q(q_val)

# ============================================================
# Generate summary-vs-q plots
# ============================================================
plot_metric_vs_q(
    metric="mem_vm_rss",
    ylabel="Final RSS Memory (GB)",
    title="Final RSS Memory vs q",
    stem="final_rss_vs_q",
    transform=kb_to_gb
)

plot_metric_vs_q(
    metric="mem_vm_size",
    ylabel="Final VM Size (GB)",
    title="Final VM Size vs q",
    stem="final_vm_size_vs_q",
    transform=kb_to_gb
)

plot_metric_vs_q(
    metric="cpu_voluntary_ctxt_switches",
    ylabel="Final Voluntary Context Switches",
    title="Final Voluntary Context Switches vs q",
    stem="final_cpu_voluntary_vs_q"
)

plot_metric_vs_q(
    metric="cpu_nonvoluntary_ctxt_switches",
    ylabel="Final Non-voluntary Context Switches",
    title="Final Non-voluntary Context Switches vs q",
    stem="final_cpu_nonvoluntary_vs_q"
)

plot_metric_vs_q(
    metric="disk_rss_file",
    ylabel="Final File-backed RSS (GB)",
    title="Final File-backed RSS vs q",
    stem="final_disk_file_vs_q",
    transform=kb_to_gb
)

plot_metric_vs_q(
    metric="disk_rss_shmem",
    ylabel="Final Shared Memory RSS (GB)",
    title="Final Shared Memory RSS vs q",
    stem="final_disk_shmem_vs_q",
    transform=kb_to_gb
)

plot_metric_vs_q(
    metric="overhead_round_total_sec",
    ylabel="Final Total Round Overhead (s)",
    title="Final Total Round Overhead vs q",
    stem="final_round_overhead_vs_q"
)

plot_metric_vs_q(
    metric="overhead_detection_sec",
    ylabel="Final Detection Overhead (s)",
    title="Final Detection Overhead vs q",
    stem="final_detection_overhead_vs_q"
)

plot_metric_vs_q(
    metric="detection_fraction",
    ylabel="Final Detection / Total Overhead",
    title="Final Detection Cost Ratio vs q",
    stem="final_detection_fraction_vs_q"
)

print(f"Saved figures to: {OUTDIR.resolve()}")
print(f"q values plotted: {q_values}")