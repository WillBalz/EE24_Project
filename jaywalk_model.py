"""
jaywalk_sim.py
──────────────
Simulates pedestrian jaywalking using an exponential inter-arrival model
where each jaywalking event accelerates future jaywalking via a user-chosen
λ-decay function.

Model:
  λₙ = λ₀ · f(n)          where f is chosen by the user
  inter-arrival time ~ Exponential(mean = λₙ)

Decay options:
  1. Exponential  : f(n) = 1 / b^n
  2. Power law    : f(n) = 1 / (n+1)^k
  3. Linear       : f(n) = 1 / (1 + c·n)
  4. Custom       : f(n) = user-typed Python expression in x  (x = n)

Usage:
  python jaywalk_sim.py
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# safe namespace available inside custom expressions
_EXPR_NS = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_EXPR_NS["np"] = np


# ─── helpers ────────────────────────────────────────────────────────────────

def prompt_float(msg, min_val=None, max_val=None):
    while True:
        try:
            val = float(input(msg))
            if min_val is not None and val < min_val:
                print(f"  Please enter a value ≥ {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"  Please enter a value ≤ {max_val}.")
                continue
            return val
        except ValueError:
            print("  Invalid input — please enter a number.")


def prompt_int(msg, min_val=None):
    while True:
        try:
            val = int(input(msg))
            if min_val is not None and val < min_val:
                print(f"  Please enter a value ≥ {min_val}.")
                continue
            return val
        except ValueError:
            print("  Invalid input — please enter a whole number.")


def quarter_label(q):
    return ["1st quarter", "2nd quarter", "3rd quarter", "4th quarter"][q]


# ─── decay-function builder ──────────────────────────────────────────────────

def build_decay_fn():
    """
    Interactively ask the user which decay model to use.
    Returns (decay_fn, label_str, params_dict).

    decay_fn(n) -> float   multiplier on λ₀  (should equal 1.0 at n=0)
    """
    print("\n  How should λ (mean wait between jaywalkers) decay over events?")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  1. Exponential  λₙ = λ₀ / b^n                          │")
    print("  │  2. Power law    λₙ = λ₀ / (n+1)^k                      │")
    print("  │  3. Linear       λₙ = λ₀ / (1 + c·n)                    │")
    print("  │  4. Custom       λₙ = λ₀ · f(x)  (you write f in Python)│")
    print("  └─────────────────────────────────────────────────────────┘")

    while True:
        choice = input("  Enter 1 / 2 / 3 / 4: ").strip()
        if choice in ("1", "2", "3", "4"):
            break
        print("  Please enter 1, 2, 3, or 4.")

    if choice == "1":
        b = prompt_float(
            "  Exponential base b  (e.g. 2 → λ halves each event, must be > 1): ",
            min_val=1.0,
        )
        decay_fn = lambda n: 1.0 / (b ** n)
        label    = f"Exponential  λₙ = λ₀ / {b}ⁿ"
        params   = {"base b": b}

    elif choice == "2":
        k = prompt_float(
            "  Power-law exponent k  (e.g. 2 → quadratic decay, must be > 0): ",
            min_val=0.001,
        )
        decay_fn = lambda n: 1.0 / ((n + 1) ** k)
        label    = f"Power law    λₙ = λ₀ / (n+1)^{k}"
        params   = {"exponent k": k}

    elif choice == "3":
        c = prompt_float(
            "  Linear rate c  (e.g. 1 → λ halves after 1st event, must be > 0): ",
            min_val=0.001,
        )
        decay_fn = lambda n: 1.0 / (1.0 + c * n)
        label    = f"Linear       λₙ = λ₀ / (1 + {c}·n)"
        params   = {"rate c": c}

    else:  # custom
        print("\n  Enter a Python expression for f(x), where:")
        print("    x  = number of jaywalkers so far (starts at 0)")
        print("    λₙ = λ₀ · f(x)   so f(0) should equal 1")
        print("    math functions available: sqrt, log, exp, sin, cos, …")
        print("    Example: 1 / (x + 1)**2")
        print("    Example: exp(-0.5 * x)")
        print("    Example: 1 / (1 + log(x + 1))")

        expr_str = None
        while True:
            raw = input("  f(x) = ").strip()
            if not raw:
                print("  Expression cannot be empty.")
                continue
            # validate: try evaluating at x=0 and x=5
            try:
                ns = dict(_EXPR_NS)
                ns["x"] = 0
                v0 = float(eval(raw, {"__builtins__": {}}, ns))
                ns["x"] = 5
                _  = float(eval(raw, {"__builtins__": {}}, ns))
            except Exception as e:
                print(f"  Could not evaluate expression: {e}")
                print("  Please try again.")
                continue

            if not math.isfinite(v0):
                print(f"  Warning: f(0) = {v0}. The expression is undefined at x=0.")
                print("  Consider using (x+1) instead of x to avoid division by zero.")
                continue

            if abs(v0 - 1.0) > 0.01:
                print(f"  Note: f(0) = {v0:.4f} (not 1.0).")
                print(f"  This means your λ₀ will be scaled by {v0:.4f} at the first event.")
                confirm = input("  Continue anyway? (y/n): ").strip().lower()
                if confirm != "y":
                    continue

            expr_str = raw
            break

        def decay_fn(n, _expr=expr_str):
            ns = dict(_EXPR_NS)
            ns["x"] = n
            return float(eval(_expr, {"__builtins__": {}}, ns))

        label  = f"Custom       λₙ = λ₀ · f(x)  where f(x) = {expr_str}"
        params = {"expression": expr_str}

    return decay_fn, label, params


# ─── simulation ─────────────────────────────────────────────────────────────

def run_simulation(n_pedestrians, lambda_0, decay_fn, decay_label,
                   decay_params, sim_duration, seed, p_0=None, alpha=None):
    rng = np.random.default_rng(seed)

    clock       = 0.0
    n_jaywalked = 0
    remaining   = n_pedestrians

    event_times    = []
    inter_arrivals = []
    lambdas_used   = []

    while remaining > 0:
        multiplier = decay_fn(n_jaywalked)
        lam_n      = lambda_0 * multiplier
        if lam_n <= 0:
            print("    λ reached zero or below — stopping simulation.")
            end_reason = "lambda_zero"
            break
        lambdas_used.append(lam_n)

        wait       = rng.exponential(scale=lam_n)
        next_clock = clock + wait

        if next_clock > sim_duration:
            end_reason = "time_limit"
            break

        clock = next_clock
        event_times.append(clock)
        inter_arrivals.append(wait)
        n_jaywalked += 1
        remaining   -= 1
    else:
        end_reason = "pool_exhausted"

    return {
        "event_times":    event_times,
        "inter_arrivals": inter_arrivals,
        "lambdas_used":   lambdas_used,
        "end_reason":     end_reason,
        "n_pedestrians":  n_pedestrians,
        "lambda_0":       lambda_0,
        "decay_label":    decay_label,
        "decay_params":   decay_params,
        "sim_duration":   sim_duration,
        "seed":           seed,
    }


# ─── stats ───────────────────────────────────────────────────────────────────

def compute_stats(sim):
    ia  = sim["inter_arrivals"]
    et  = sim["event_times"]
    dur = sim["sim_duration"]
    n_p = sim["n_pedestrians"]

    n_jaywalked  = len(et)
    avg_inter    = float(np.mean(ia)) if ia else 0.0
    jaywalk_prob = n_jaywalked / n_p  if n_p > 0 else 0.0

    quarter_counts = [0, 0, 0, 0]
    for t in et:
        q = min(int(t / dur * 4), 3)
        quarter_counts[q] += 1

    return {
        "n_jaywalked":    n_jaywalked,
        "avg_inter":      avg_inter,
        "jaywalk_prob":   jaywalk_prob,
        "quarter_counts": quarter_counts,
        "median_inter":   float(np.median(ia)) if ia else 0.0,
        "std_inter":      float(np.std(ia))    if ia else 0.0,
        "min_inter":      float(np.min(ia))    if ia else 0.0,
        "max_inter":      float(np.max(ia))    if ia else 0.0,
    }


# ─── binomial expected curve ─────────────────────────────────────────────────

def compute_binomial_expected(sim):
    """
    At each jaywalking event n, every still-waiting pedestrian faces a
    temptation trial with probability:

        p_n = min(1, p_0 * (1 + alpha * n))

    Expected remaining pedestrians evolves as:
        E[remaining_0] = N
        E[remaining_{n+1}] = E[remaining_n] * (1 - p_n)

    Expected cumulative jaywalkers after trial n:
        E[cumulative_n] = N - E[remaining_n]

    Returns
    -------
    expected_cumulative : list[float]  one value per simulation event
    p_series            : list[float]  p_n at each trial  (for plotting)
    """
    n_p      = sim["n_pedestrians"]
    p_0      = sim["p_0"]
    alpha    = sim["alpha"]
    n_events = len(sim["event_times"])

    expected_cumulative = []
    p_series            = []
    e_remaining         = float(n_p)

    for n in range(n_events):
        p_n = min(1.0, p_0 * (1.0 + alpha * n))
        p_series.append(p_n)
        e_remaining *= (1.0 - p_n)
        expected_cumulative.append(n_p - e_remaining)

    return expected_cumulative, p_series



def print_report(sim, stats):
    sep = "─" * 56
    print(f"\n{'═'*56}")
    print("  JAYWALKING SIMULATION REPORT")
    print(f"{'═'*56}")

    print("\n📋  INPUTS")
    print(sep)
    print(f"  Total pedestrians waiting  : {sim['n_pedestrians']}")
    print(f"  Initial mean wait λ₀       : {sim['lambda_0']} s")
    print(f"  Decay model                : {sim['decay_label']}")
    for k, v in sim["decay_params"].items():
        print(f"    {k:28s}: {v}")
    print(f"  Simulation duration        : {sim['sim_duration']} s")
    print(f"  Random seed                : {sim['seed']}")

    print("\n📊  RESULTS")
    print(sep)
    end_tags = {
        "time_limit":      "⏱  time limit reached",
        "pool_exhausted":  "pool exhausted early",
        "lambda_zero":     "⚠  λ decayed to zero",
    }
    print(f"  End condition              : {end_tags.get(sim['end_reason'], sim['end_reason'])}")
    print(f"  Total jaywalkers           : {stats['n_jaywalked']}")
    print(f"  Pedestrians who did NOT    : {sim['n_pedestrians'] - stats['n_jaywalked']}")
    print(f"  Jaywalk probability        : {stats['jaywalk_prob']:.1%}")

    print("\n⏱  INTER-ARRIVAL TIMES (seconds)")
    print(sep)
    print(f"  Mean                       : {stats['avg_inter']:.3f} s")
    print(f"  Median                     : {stats['median_inter']:.3f} s")
    print(f"  Std dev                    : {stats['std_inter']:.3f} s")
    print(f"  Min                        : {stats['min_inter']:.3f} s")
    print(f"  Max                        : {stats['max_inter']:.3f} s")

    print("\n🕐  JAYWALKERS PER QUARTER")
    print(sep)
    for i, count in enumerate(stats["quarter_counts"]):
        pct = count / stats["n_jaywalked"] * 100 if stats["n_jaywalked"] else 0
        print(f"  {quarter_label(i):12s}  : {count:4d}  ({pct:5.1f}%)")

    print(f"\n{'═'*56}\n")


# ─── plots ────────────────────────────────────────────────────────────────────

def make_plots(sim, stats, binom_expected, p_series):
    et   = sim["event_times"]
    ia   = sim["inter_arrivals"]
    lams = sim["lambdas_used"]
    dur  = sim["sim_duration"]
    n_p  = sim["n_pedestrians"]

    C1, C2, C3, C4, C5 = "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#9B5DE5"
    BG   = "#F8F9FA"
    GRID = "#DEE2E6"

    fig = plt.figure(figsize=(16, 18), facecolor=BG)
    fig.suptitle(
        f"Jaywalking Simulation — {sim['decay_label']}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    # ── 1. Cumulative jaywalkers + remaining ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_facecolor(BG)
    ax1.grid(color=GRID, linewidth=0.8, zorder=0)

    if et:
        times_plot     = [0] + et
        counts         = list(range(len(times_plot)))
        remaining_plot = [n_p - c for c in counts]

        ax1.step(times_plot, counts, where="post", color=C1, linewidth=2,
                 label="Cumulative jaywalkers", zorder=3)
        ax1.fill_between(times_plot, counts, step="post",
                         color=C1, alpha=0.15, zorder=2)
        ax1.step(times_plot, remaining_plot, where="post", color=C2,
                 linewidth=2, linestyle="--", label="Remaining waiting", zorder=3)

    for q in range(1, 4):
        ax1.axvline(dur * q / 4, color=GRID, linewidth=1.2, linestyle=":", zorder=1)
        ax1.text(dur * q / 4, 0.5, f"Q{q}", color="#868E96",
                 fontsize=8, ha="center", transform=ax1.get_xaxis_transform())

    ax1.axvline(dur, color="#6C757D", linewidth=1, zorder=1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("People")
    ax1.set_title("Cumulative Jaywalkers & Remaining Pedestrians")
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 2. Jaywalkers per quarter ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG)
    ax2.grid(color=GRID, axis="y", linewidth=0.8, zorder=0)

    qc    = stats["quarter_counts"]
    bars  = ax2.bar(["Q1","Q2","Q3","Q4"], qc,
                    color=[C1,C2,C3,C4], zorder=3,
                    edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, qc):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.2, str(val),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_title("Jaywalkers per Quarter")
    ax2.set_ylabel("Count")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 3. Inter-arrival histogram ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG)
    ax3.grid(color=GRID, linewidth=0.8, zorder=0)

    if ia:
        n_bins = max(10, min(40, len(ia) // 3 + 1))
        ax3.hist(ia, bins=n_bins, color=C3, edgecolor="white",
                 linewidth=0.6, zorder=3, density=True, label="Observed")

        x_max = max(ia) * 1.1
        xs    = np.linspace(0, x_max, 300)
        mu    = stats["avg_inter"]
        if mu > 0:
            pdf = (1 / mu) * np.exp(-xs / mu)
            ax3.plot(xs, pdf, color=C1, linewidth=2, linestyle="--",
                     label=f"Exp(μ={mu:.2f}s) PDF")
        ax3.axvline(mu, color=C1, linewidth=1.5, linestyle=":",
                    label=f"Mean = {mu:.2f}s")
        ax3.legend(fontsize=8)

    ax3.set_title("Distribution of Inter-Arrival Times")
    ax3.set_xlabel("Time between jaywalkers (s)")
    ax3.set_ylabel("Density")

    # ── 4. λ decay over successive events ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG)
    ax4.grid(color=GRID, linewidth=0.8, zorder=0)

    if lams:
        xs_l = list(range(len(lams)))
        ax4.plot(xs_l, lams, color=C2, linewidth=2,
                 marker="o", markersize=4, zorder=3, label="Simulated λₙ")
        ax4.fill_between(xs_l, lams, color=C2, alpha=0.15, zorder=2)
        ax4.legend(fontsize=8)

    ax4.set_title("λ (Mean Wait) per Jaywalking Event")
    ax4.set_xlabel("Event number (n)")
    ax4.set_ylabel("λₙ (seconds)")
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 5. Instantaneous rate over time ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(BG)
    ax5.grid(color=GRID, linewidth=0.8, zorder=0)

    if et and len(et) >= 2:
        rates = [1.0 / l for l in lams[:len(et)]]
        ax5.step(et, rates, where="pre", color=C4, linewidth=2, zorder=3)
        ax5.fill_between(et, rates, step="pre", color=C4, alpha=0.2, zorder=2)
        ax5.set_xlim(0, dur)
    else:
        ax5.text(0.5, 0.5, "Not enough events\nto plot rate",
                 ha="center", va="center", transform=ax5.transAxes, color="#868E96")

    ax5.set_title("Instantaneous Jaywalking Rate Over Time")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Rate (jaywalkers / s)")

    # ── 6. Binomial expected vs simulation actual (wide, full bottom-left) ──
    ax6 = fig.add_subplot(gs[2, 0:2])
    ax6.set_facecolor(BG)
    ax6.grid(color=GRID, linewidth=0.8, zorder=0)

    if et:
        # actual simulation curve
        times_plot = [0] + et
        counts     = list(range(len(times_plot)))
        ax6.step(times_plot, counts, where="post", color=C1, linewidth=2,
                 label="Simulation (actual)", zorder=3)
        ax6.fill_between(times_plot, counts, step="post",
                         color=C1, alpha=0.10, zorder=2)

        # binomial expected curve — plotted at each event time
        if binom_expected:
            binom_times = [0] + et
            binom_vals  = [0] + binom_expected
            ax6.plot(binom_times, binom_vals, color=C5, linewidth=2.5,
                     linestyle="--", marker="o", markersize=3,
                     label="Binomial model (expected)", zorder=4)

        # divergence shading between the two curves
        # interpolate to a common fine grid for fill
        if binom_expected and len(et) >= 2:
            sim_y   = np.array([0] + list(range(1, len(et) + 1)), dtype=float)
            binom_y = np.array([0] + binom_expected, dtype=float)
            t_arr   = np.array([0] + et)
            ax6.fill_between(t_arr, sim_y, binom_y,
                             where=(sim_y >= binom_y),
                             color=C1, alpha=0.18, zorder=1,
                             label="Sim ahead of model")
            ax6.fill_between(t_arr, sim_y, binom_y,
                             where=(sim_y < binom_y),
                             color=C5, alpha=0.18, zorder=1,
                             label="Model ahead of sim")

    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Cumulative jaywalkers")
    ax6.set_title(
        f"Binomial Model vs Simulation  "
        f"(p₀={sim['p_0']}, α={sim['alpha']})"
    )
    ax6.legend(fontsize=9)
    ax6.yaxis.set_major_locator(MaxNLocator(integer=True))

    # ── 7. p_n growth over trials ───────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(BG)
    ax7.grid(color=GRID, linewidth=0.8, zorder=0)

    if p_series:
        ax7.plot(range(len(p_series)), p_series, color=C5, linewidth=2,
                 marker="o", markersize=4, zorder=3)
        ax7.fill_between(range(len(p_series)), p_series,
                         color=C5, alpha=0.15, zorder=2)
        ax7.axhline(1.0, color=GRID, linewidth=1, linestyle="--")
        ax7.set_ylim(0, min(1.1, max(p_series) * 1.3))

    ax7.set_title("Trial Probability pₙ Growth (Social Contagion)")
    ax7.set_xlabel("Event number (n)")
    ax7.set_ylabel("pₙ")
    ax7.xaxis.set_major_locator(MaxNLocator(integer=True))

    out_path = os.path.join(SCRIPT_DIR, "jaywalk_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  📈  Figure saved to {out_path}")
    plt.show()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 56)
    print("  JAYWALKING SIMULATION")
    print("  Exponential inter-arrival model with user-chosen λ-decay")
    print("═" * 56 + "\n")

    n_pedestrians = prompt_int(
        "1. Total pedestrians waiting to cross: ",
        min_val=1,
    )
    lambda_0 = prompt_float(
        "2. Initial mean time between jaywalkers λ₀ (seconds): ",
        min_val=0.001,
    )

    decay_fn, decay_label, decay_params = build_decay_fn()

    sim_duration = prompt_float(
        "\n3. Simulation duration (seconds): ",
        min_val=0.001,
    )
    seed_input = input("4. Random seed for reproducibility (leave blank for random): ").strip()
    seed = int(seed_input) if seed_input.lstrip("-").isdigit() else None

    print("\n  Running simulation …")
    sim   = run_simulation(n_pedestrians, lambda_0, decay_fn, decay_label,
                           decay_params, sim_duration, seed, p_0=None, alpha=None)
    stats = compute_stats(sim)
    print_report(sim, stats)

    # ── binomial overlay (separate from simulation) ──────────────────────
    print("─" * 56)
    print("  BINOMIAL MODEL OVERLAY")
    print("  This is independent of the simulation above.")
    print("  We'll compute what a binomial model would predict")
    print("  and plot it against your simulation results.")
    print("  Each time someone jaywalks, every waiting pedestrian")
    print("  faces a trial:  pₙ = min(1, p₀ · (1 + α · n))")
    print("─" * 56)
    p_0 = prompt_float(
        "p₀  base trial probability  (e.g. 0.05 = 5% per trial at start): ",
        min_val=0.0, max_val=1.0,
    )
    alpha = prompt_float(
        "α   contagion growth        (0 = fixed p, 0.1 = grows 10% per event): ",
        min_val=0.0,
    )

    sim["p_0"]   = p_0
    sim["alpha"] = alpha
    binom_expected, p_series = compute_binomial_expected(sim)

    make_plots(sim, stats, binom_expected, p_series)


if __name__ == "__main__":
    main()