"""Generate the Jupyter notebook companion from the tutorial markdown.

The tutorial (`tutorial_covered_call_backtest.md`) is the single source of
truth. This script parses it into notebook cells so the two never drift:

  * Prose becomes markdown cells, split at H2/H3 headings so each cell is a
    digestible section a reader can run-as-they-read.
  * Fenced ``python`` blocks that actually compile become runnable code
    cells. Signature stubs and pseudo-code (the tutorial's "illustrative,
    not in the codebase" excerpts) don't compile, so they stay rendered as
    markdown — exactly as they read in the tutorial.
  * Every ``![...](docs/figures/NN_*.png)`` embed is replaced by the
    matching chart-generation call from `make_figures.py`, so the chart
    code is visible and the figure renders inline when the cell runs.
  * Sections that *link* to code instead of inlining it (so there's no
    fenced block to convert) get a generated demo cell that runs the
    linked helpers — see ``LINKED_CODE_DEMOS``.

A setup cell at the top clones the repo and pip-installs when running on
Google Colab (no-op locally), then imports the engine's public API.

Run: python make_notebook.py   →   covered_call_backtest.ipynb

CI/consistency note: this is the notebook's only source. Re-run it after
editing the tutorial or the figure script and commit the regenerated
.ipynb (the same contract `make_figures.py` has with the PNGs).
"""

from __future__ import annotations

import json
import re

TUTORIAL = "tutorial_covered_call_backtest.md"
OUT = "covered_call_backtest.ipynb"

# docs/figures/NN_name.png  ->  the make_figures call that regenerates it.
# stats / summary / daily_equity are bound by the data-prep cell below.
FIGURE_CALLS: dict[str, str] = {
    "01_equity_curves.png": "fig1_equity_curves(daily_equity, summary)",
    "02_excess_histogram.png": "fig2_excess_histogram(daily_equity, summary, stats)",
    "03_bias_variance.png": "fig3_bias_variance()",
    "04_t_stat_vs_years.png": 'fig4_t_stat_vs_years(stats["sharpe_excess"])',
}

IMAGE_RE = re.compile(r"^!\[.*\]\((?:\./)?docs/figures/([0-9A-Za-z_]+\.png)\)\s*$")

# Some tutorial sections *link* to code in cc_backtest.py instead of inlining
# it (the repo's "linked, test-pinned implementations" convention), so the
# converter sees no fenced block to turn into a runnable cell. For those, map
# the exact section heading -> a short demo that exercises the linked helpers
# on the bundled MSFT data. The demo cell is emitted right after the section's
# prose. Same single-source idea as FIGURE_CALLS: the runnable code lives here
# in the generator, not in the tutorial markdown.
LINKED_CODE_DEMOS: dict[str, str] = {
    "### The IV Proxy: Why a Regime-Based Multiplier Works": '''\
# Run the linked helpers on the bundled MSFT data:
#   calc_rolling_volatility  ->  detect_regime  ->  estimate_iv
# (the three cc_backtest.py functions this section links to)
import collections

rolling_vol = calc_rolling_volatility(prices, window=30)

# Most recent day with a valid (non-NaN warm-up) rolling vol
i = int(np.flatnonzero(~np.isnan(rolling_vol))[-1])
hv = float(rolling_vol[i])
regime = detect_regime(hv)
iv = estimate_iv(hv)
print(
    f"{dates[i]}  30-day HV {hv:6.2%}  ->  regime {regime:<6}"
    f"  ->  IV estimate {iv:6.2%}  ({iv / hv:.1f}x)"
)

# Regime mix across the whole sample (multipliers: high 1.1 / normal 1.3 / low 1.5)
valid = rolling_vol[~np.isnan(rolling_vol)]
mix = collections.Counter(detect_regime(float(v)) for v in valid)
for r in ("low", "normal", "high"):
    print(f"  {r:<6} {mix[r]:4d} days ({mix[r] / len(valid):5.1%})")''',
    "### The State Machine: OPEN → Check → Handle → Reset": '''\
# The real engine: run_cc_overlay inlines exactly this state machine.
# (run_cc_overlay_day above is a teaching sketch — never called by the codebase.)
import collections

summary, trades, _ = run_cc_overlay(dates, prices, params)

# The engine's trade actions map 1:1 onto the diagram's branches:
#   sell        IDLE -> OPEN: sold a 0.25-delta call
#   close       profit target hit (75% of premium captured)
#   close_itm   deep-ITM assignment risk (delta > 0.70)
#   expiration  reached expiry: assigned if ITM, else expired worthless
counts = collections.Counter(t["action"] for t in trades)
for action in ("sell", "close", "close_itm", "expiration"):
    print(f"  {action:<11} {counts[action]:4d}")

print("\\nFirst 4 trade-state transitions:")
for t in trades[:4]:
    extra = (
        f" strike ${t['strike']:.0f}"
        if t["action"] == "sell"
        else f" pnl ${t['pnl']:+.2f}"
    )
    print(f"  {t['date']}  {t['action']:<11}{extra}")''',
}

SETUP_CODE = '''\
# === Setup — runs anywhere; clones + installs only on Google Colab ===
import os
import subprocess
import sys

if "google.colab" in sys.modules:
    if not os.path.isdir("covered-call-backtesting"):
        subprocess.run(
            ["git", "clone",
             "https://github.com/l3a0/covered-call-backtesting.git"],
            check=True,
        )
    os.chdir("covered-call-backtesting")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )

%matplotlib inline
import numpy as np

from cc_backtest import (
    bs_delta,
    bs_price,
    calc_rolling_volatility,
    compute_statistics,
    detect_regime,
    estimate_iv,
    find_strike_for_delta,
    normal_cdf,
    normal_pdf,
    run_cc_overlay,
    walk_forward_optimization,
)
'''

DATA_PREP_CODE = '''\
# === Run the backtest once — the figure cells below reuse these results ===
from make_figures import (
    fig1_equity_curves,
    fig2_excess_histogram,
    fig3_bias_variance,
    fig4_t_stat_vs_years,
    load_msft_csv,
)

dates, prices = load_msft_csv("msft_10yr_prices.csv")
params = {
    "call_delta": 0.25,
    "close_at_pct": 0.75,
    "dte": 21,
    "risk_free_rate": 0.045,
    "capital": 100_000,
}
summary, _trades, daily_equity = run_cc_overlay(dates, prices, params)
stats = compute_statistics(
    daily_equity,
    num_contracts=summary["num_contracts"],
    cash=summary["cash"],
)
print(
    f"Backtest ready — Sharpe of excess return: {stats['sharpe_excess']:+.3f}, "
    f"Newey-West t-stat: {stats['t_stat_newey_west']:+.2f}"
)
'''

INTRO_MD = '''\
# Covered Call Backtester — Notebook Companion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/l3a0/covered-call-backtesting/blob/main/covered_call_backtest.ipynb)

This notebook is generated from
[`tutorial_covered_call_backtest.md`](https://github.com/l3a0/covered-call-backtesting/blob/main/tutorial_covered_call_backtest.md)
by [`make_notebook.py`](https://github.com/l3a0/covered-call-backtesting/blob/main/make_notebook.py)
— the tutorial is the source of truth; don't hand-edit this file.

Run the two setup cells first, then read top-to-bottom. Code cells are the
tutorial's runnable excerpts plus the chart-generation calls from
`make_figures.py`; signature stubs and pseudo-code stay as formatted text,
just as they appear in the tutorial.
'''


def compiles(src: str) -> bool:
    """True if the snippet is a syntactically valid module on its own.

    Signature-only stubs and pseudo-code (top-level ``return``) raise
    SyntaxError and stay rendered as markdown, which is how the tutorial
    presents them anyway.
    """
    if not src.strip():
        return False
    try:
        compile(src, "<tutorial-cell>", "exec")
    except SyntaxError:
        return False
    return True


def _src(text: str) -> list[str]:
    """nbformat source list: each line keeps its newline except the last."""
    lines = text.splitlines()
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:] if lines else []


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(text),
    }


def build_cells(md: str) -> list[dict]:
    lines = md.split("\n")
    cells: list[dict] = [md_cell(INTRO_MD), code_cell(SETUP_CODE),
                         code_cell(DATA_PREP_CODE)]
    buf: list[str] = []
    pending_demo: str | None = None

    def flush() -> None:
        text = "\n".join(buf).strip("\n")
        if text.strip():
            cells.append(md_cell(text))
        buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            lang = line[3:].strip()
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            src = "\n".join(block)
            if lang == "python" and compiles(src):
                flush()
                cells.append(code_cell(src))
            else:
                buf.append("```" + lang)
                buf.extend(block)
                buf.append("```")
            continue

        m = IMAGE_RE.match(line.strip())
        if m and m.group(1) in FIGURE_CALLS:
            flush()
            call = FIGURE_CALLS[m.group(1)]
            cells.append(code_cell(f"# Regenerates docs/figures/{m.group(1)}\n_ = {call}"))
            i += 1
            continue

        if line.startswith("## ") or line.startswith("### "):
            if any(b.strip() for b in buf):
                flush()
            # The just-finished section ended here — emit its demo (if any)
            # after its prose, before the new heading starts accumulating.
            if pending_demo is not None:
                cells.append(code_cell(pending_demo))
                pending_demo = None
            if line in LINKED_CODE_DEMOS:
                pending_demo = LINKED_CODE_DEMOS[line]
        buf.append(line)
        i += 1

    flush()
    if pending_demo is not None:  # demo on the final section
        cells.append(code_cell(pending_demo))
    return cells


def main() -> None:
    with open(TUTORIAL, encoding="utf-8") as f:
        md = f.read()

    notebook = {
        "cells": build_cells(md),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")

    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
    print(f"Wrote {OUT}: {len(notebook['cells'])} cells ({n_code} code, {n_md} markdown)")


if __name__ == "__main__":
    main()
