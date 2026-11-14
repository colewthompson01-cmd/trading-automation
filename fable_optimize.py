#!/usr/bin/env python3
"""
fable_optimize.py — One-time Fable session to optimize 0DTE scanner signal weights.

Reads backtest_results.csv, computes signal accuracy, sends full context to
claude-fable-5-2, and writes the optimized build_recommendation() to
fable_optimized_weights.py.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 fable_optimize.py
"""

import os
import sys
import json
import subprocess

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import anthropic
except ImportError:
    print("Installing anthropic SDK...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])
    import anthropic

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

# ── API key check ─────────────────────────────────────────────────────────────
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("\n  ❌  ANTHROPIC_API_KEY is not set.")
    print("  Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
    print("  Then re-run this script.\n")
    sys.exit(1)

# ── Load backtest data ────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "backtest_results.csv")
if not os.path.exists(CSV_PATH):
    print(f"  ❌  {CSV_PATH} not found. Run backtest_0dte.py --save first.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
go_df = df[df["go_nogo"].str.startswith("GO")].copy()

print(f"  Loaded {len(df)} rows, {len(go_df)} GO-threshold trades")

# ── Compute signal accuracy (GO trades only) ──────────────────────────────────
sig_cols = [c for c in go_df.columns if c.startswith("sig_")]
signal_stats = []
for col in sig_cols:
    label = col.replace("sig_", "").replace("_", " ")
    sub = go_df[go_df[col] != 0].copy()
    sub = sub[sub["outcome"].isin(["WIN", "LOSS"])]
    if len(sub) < 20:
        continue
    correct = (
        ((sub[col] > 0) & (sub["outcome"] == "WIN")) |
        ((sub[col] < 0) & (sub["outcome"] == "LOSS"))
    ).mean() * 100
    signal_stats.append((label, round(correct, 1), len(sub)))

signal_stats.sort(key=lambda x: x[1], reverse=True)
signal_summary = {label: {"accuracy_pct": acc, "sample_size": n}
                  for label, acc, n in signal_stats}

# ── Compute regime and ticker win rates ───────────────────────────────────────
regime_stats = {}
for regime in ["LOW (<15)", "NORMAL (15-28)", "HIGH (>28)"]:
    sub = go_df[go_df["vix_regime"] == regime]
    if len(sub) == 0:
        regime_stats[regime] = {"win_pct": None, "n_trades": 0}
    else:
        regime_stats[regime] = {
            "win_pct": round((sub["outcome"] == "WIN").mean() * 100, 1),
            "n_trades": len(sub),
        }

ticker_stats = {}
for tk, grp in go_df.groupby("ticker"):
    ticker_stats[tk] = {
        "win_pct": round((grp["outcome"] == "WIN").mean() * 100, 1),
        "n_trades": len(grp),
    }
ticker_stats = dict(sorted(ticker_stats.items(), key=lambda x: x[1]["win_pct"], reverse=True))

direction_stats = {}
for d in ["CALL", "PUT"]:
    sub = go_df[go_df["direction"] == d]
    direction_stats[d] = {
        "win_pct": round((sub["outcome"] == "WIN").mean() * 100, 1) if len(sub) else None,
        "n_trades": len(sub),
    }

overall_win = round((go_df["outcome"] == "WIN").mean() * 100, 1)

# ── Current build_recommendation() source ────────────────────────────────────
SCANNER_PATH = os.path.join(os.path.dirname(__file__), "0dte_scanner.py")
with open(SCANNER_PATH) as f:
    scanner_lines = f.readlines()

# Extract build_recommendation function
start = next(i for i, l in enumerate(scanner_lines) if "def build_recommendation(" in l)
end = next(i for i in range(start + 1, len(scanner_lines))
           if scanner_lines[i].startswith("def ") or scanner_lines[i].startswith("# ──"))
current_fn = "".join(scanner_lines[start:end])

# ── Build prompt ──────────────────────────────────────────────────────────────
DATA_JSON = json.dumps({
    "backtest_period": "2 years (Nov 2024 – Nov 2026)",
    "total_tickers": 14,
    "total_rows": len(df),
    "go_threshold_trades": len(go_df),
    "overall_win_rate_pct": overall_win,
    "signal_accuracy_on_go_trades": signal_summary,
    "vix_regime_win_rates": regime_stats,
    "ticker_win_rates_go_trades": ticker_stats,
    "direction_win_rates": direction_stats,
    "key_observations": [
        "Only 417 / 3585 trades crossed the GO threshold (conviction >= 5)",
        "All GO trades fall in NORMAL VIX (15-28) — VIX penalty eliminates LOW/HIGH regimes",
        "Overall win rate is 37.2% — below break-even for 0DTE options at typical IV",
        "VIX Direction is the strongest individual predictor at 57.4%",
        "Sector Pulse and DXY are second-tier at 56%",
        "CALL trades (40.9% win) outperform PUT trades (31.5% win)",
        "Best tickers: MU (62.1%), TSLA (58.1%), AMD (51.5%)",
        "Worst tickers: QQQ (21.4%), AAPL (22.2%), SPY (24.1%)",
        "Sentiment (News Sentiment) did not appear in signal_accuracy — NEUTRAL was forced in backtest",
    ],
}, indent=2)

USER_PROMPT = f"""You are a quantitative trading systems engineer specializing in 0DTE (zero days-to-expiry) options.

I have a Python-based 0DTE options scanner that uses a weighted scoring system to recommend CALL or PUT trades. I ran a 2-year backtest and need you to redesign the signal weights and scoring formula based on the empirical accuracy data.

## Current build_recommendation() function

```python
{current_fn}
```

## 2-Year Backtest Results

```json
{DATA_JSON}
```

## What I need from you

1. **Redesign the base signal weights** (the 4th tuple element in each `signals.append()` call) to reflect predictive accuracy. Currently all weights are 1 or 2. Use the accuracy data to give higher-accuracy signals more weight.

2. **Redesign the VIX regime multipliers** (`w_tech`, `w_sent`). Currently there are 3 regimes with equal scaling. The backtest shows LOW VIX (<15) yields no GO trades (VIX penalty kills conviction). Consider whether VIX penalty should be:
   - Removed entirely and replaced with a hard block
   - Kept but with different thresholds
   - Increased to be more selective in NORMAL regime

3. **Fix the conviction formula**. Currently: `conviction = abs(ratio) * 5 + cat_score + vix_pen`. With `cat_score=2` as baseline and `abs(ratio)` rarely exceeding 0.5, conviction clusters around 2-4. The GO threshold is 5. Suggest a formula that is more discriminating — creates more separation between strong and weak signals.

4. **Recommend ticker filter**. Based on win rates, should I remove any tickers from the scanner? Be specific.

5. **Suggest direction bias adjustments**. CALL win rate (40.9%) > PUT win rate (31.5%). Should PUT trades require higher conviction?

6. **Mixed-signal conflict detection**. Current rule: cap conviction at 6 when |bull_votes - bear_votes| <= 1 AND total_votes >= 4. Is this too aggressive? Too lenient?

## Output format

Provide:
1. A fully rewritten `build_recommendation()` function with your optimized weights inline
2. A list of tickers to remove (if any)
3. A short explanation (5-7 bullet points) of the key changes you made and why
4. Any additional rules or thresholds you added

The function signature must stay the same: `build_recommendation(tech, cross, sectors, sentiment, cat_score, analyst_net=0)`
Return signature must stay: `return direction, conviction, go_nogo, signals`

Write production-ready Python code I can drop in directly. Use the exact signal labels that already exist in the code.
"""

# ── Call Fable ────────────────────────────────────────────────────────────────
print("\n  Calling claude-fable-5-2... (this may take 30-90 seconds)\n")

client = anthropic.Anthropic()

with client.beta.messages.stream(
    model="claude-fable-5-2",
    max_tokens=32000,
    output_config={"effort": "high"},
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
    messages=[{"role": "user", "content": USER_PROMPT}],
) as stream:
    response = stream.get_final_message()

# ── Extract text response ─────────────────────────────────────────────────────
result_text = ""
for block in response.content:
    if block.type == "text":
        result_text += block.text

print("=" * 72)
print(result_text)
print("=" * 72)

# ── Save output ───────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(os.path.dirname(__file__), "fable_optimized_weights.py")
with open(OUT_PATH, "w") as f:
    f.write("# Fable-optimized build_recommendation() — generated by fable_optimize.py\n")
    f.write("# Review and copy the function below into 0dte_scanner.py\n\n")
    f.write(result_text)

print(f"\n  ✅  Response saved to {OUT_PATH}")
print(f"  Usage: {response.usage.input_tokens} in / {response.usage.output_tokens} out tokens")
print(f"  Stop reason: {response.stop_reason}")
