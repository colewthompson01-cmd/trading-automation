#!/usr/bin/env python3
"""
fable_confluence.py — Ask Fable to find high-win-rate signal combinations
and suggest new filters/signals for the 0DTE scanner.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 fable_confluence.py
"""

import os, sys, json, subprocess

try:
    import anthropic
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])
    import anthropic

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

from itertools import combinations

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("\n  ANTHROPIC_API_KEY is not set.")
    print("  Set it with:  export ANTHROPIC_API_KEY=sk-ant-...\n")
    sys.exit(1)

CSV_PATH = os.path.join(os.path.dirname(__file__), "backtest_results.csv")
if not os.path.exists(CSV_PATH):
    print(f"  {CSV_PATH} not found. Run: python3 backtest_0dte.py --years 2 --save")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
go_df = df[df["go_nogo"].str.startswith("GO") & df["outcome"].isin(["WIN", "LOSS"])].copy()
print(f"  Loaded {len(df)} rows, {len(go_df)} GO trades with outcomes")

# ── Signal columns ────────────────────────────────────────────────────────────
sig_cols = [c for c in go_df.columns if c.startswith("sig_")]
sig_labels = [c.replace("sig_", "").replace("_", " ") for c in sig_cols]

# ── Individual signal accuracy ────────────────────────────────────────────────
single_stats = {}
for col, label in zip(sig_cols, sig_labels):
    sub = go_df[go_df[col] != 0].copy()
    if len(sub) < 15:
        continue
    correct = (
        ((sub[col] > 0) & (sub["outcome"] == "WIN")) |
        ((sub[col] < 0) & (sub["outcome"] == "LOSS"))
    ).mean() * 100
    single_stats[label] = {"accuracy_pct": round(correct, 1), "n": len(sub)}

# ── Pairwise signal combinations (both signals pointing same direction) ───────
pair_stats = []
for (c1, l1), (c2, l2) in combinations(zip(sig_cols, sig_labels), 2):
    sub = go_df[(go_df[c1] != 0) & (go_df[c2] != 0) & (go_df[c1] == go_df[c2])].copy()
    if len(sub) < 15:
        continue
    win_rate = (sub["outcome"] == "WIN").mean() * 100
    pair_stats.append({
        "signals": [l1, l2],
        "win_rate_pct": round(win_rate, 1),
        "n_trades": len(sub),
    })

pair_stats.sort(key=lambda x: x["win_rate_pct"], reverse=True)

# ── Triple combinations ───────────────────────────────────────────────────────
triple_stats = []
for (c1, l1), (c2, l2), (c3, l3) in combinations(zip(sig_cols, sig_labels), 3):
    sub = go_df[
        (go_df[c1] != 0) & (go_df[c2] != 0) & (go_df[c3] != 0) &
        (go_df[c1] == go_df[c2]) & (go_df[c2] == go_df[c3])
    ].copy()
    if len(sub) < 10:
        continue
    win_rate = (sub["outcome"] == "WIN").mean() * 100
    triple_stats.append({
        "signals": [l1, l2, l3],
        "win_rate_pct": round(win_rate, 1),
        "n_trades": len(sub),
    })

triple_stats.sort(key=lambda x: x["win_rate_pct"], reverse=True)

# ── Conviction band breakdown ────────────────────────────────────────────────
conv_bands = {}
for band, lo, hi in [("7.0-7.5", 7.0, 7.5), ("7.5-8.0", 7.5, 8.0),
                     ("6.5-7.0", 6.5, 7.0), ("8.0-8.5", 8.0, 8.5)]:
    sub = go_df[(go_df["conviction"] >= lo) & (go_df["conviction"] < hi)]
    if len(sub) > 0:
        conv_bands[band] = {
            "win_pct": round((sub["outcome"] == "WIN").mean() * 100, 1),
            "n": len(sub),
        }

# ── Current ticker win rates ──────────────────────────────────────────────────
ticker_stats = {}
for tk, grp in go_df.groupby("ticker"):
    ticker_stats[tk] = {
        "win_pct": round((grp["outcome"] == "WIN").mean() * 100, 1),
        "n": len(grp),
    }
ticker_stats = dict(sorted(ticker_stats.items(), key=lambda x: x[1]["win_pct"], reverse=True))

# ── Overall stats ─────────────────────────────────────────────────────────────
overall_win = round((go_df["outcome"] == "WIN").mean() * 100, 1)

DATA = {
    "backtest_period": "2 years",
    "total_go_trades": len(go_df),
    "overall_win_rate_pct": overall_win,
    "current_signal_weights": {
        "VIX Direction": 2.0, "Sector Pulse": 1.75, "DXY": 1.75,
        "10Y Yield": 1.25, "S&P Futures": 1.0, "Pre-Market Move": 1.0,
        "RSI (14)": 0.5, "BB %B": 0.5, "News Sentiment": 0.5,
        "Analyst": 0.25, "VIX IVR": 0.75, "Put/Call Ratio": 0.75,
    },
    "individual_signal_accuracy": single_stats,
    "top_20_pair_combinations": pair_stats[:20],
    "top_20_triple_combinations": triple_stats[:20],
    "conviction_band_win_rates": conv_bands,
    "ticker_win_rates": ticker_stats,
    "scanner_tickers": ["NVDA", "TSLA", "META", "AMZN", "AMD", "MU", "NFLX", "MSTR", "HOOD"],
    "go_window": "conviction 7.0 to 8.0 (above 8.0 = over-signaled, win rate drops to 44%)",
    "data_source": "yfinance daily OHLCV — no intraday, no options flow, no order book",
}

PROMPT = f"""You are a quantitative trading systems engineer specializing in 0DTE (zero days-to-expiry) options strategies.

I have a Python 0DTE scanner that generates GO/NO-GO signals using weighted market indicators. It currently achieves a 51.5% win rate on backtest. I want to push toward 60-65%+ by finding the highest-accuracy signal combinations and identifying new signals worth adding.

## Current System
- Signals: VIX Direction, Sector Pulse, DXY, 10Y Yield, S&P Futures, Pre-Market Move, RSI(14), BB%B, News Sentiment, VIX IVR, Put/Call Ratio
- GO only when: conviction in [7.0, 8.0] (conviction = 10 × agreement × √participation + catalyst_adj)
- Data source: yfinance daily OHLCV (free, backtestable) — no intraday, no paid feeds
- Scanner runs at 10:00 AM CST on market days

## 2-Year Backtest Data

```json
{json.dumps(DATA, indent=2)}
```

## Questions

### 1. High-Win-Rate Signal Combinations
From the pair and triple combination data above, which specific combinations have the best balance of win rate AND trade frequency? I need at least 8-10 trades/year to be useful.

If I were to add a "super filter" — only GO when these specific signals all agree — what win rate could I realistically expect, and how many trades per year would I get?

### 2. New Signals to Add (backtestable with yfinance)
What additional signals could meaningfully improve win rate? Focus on signals I can compute from yfinance daily data:
- Examples: MACD, stochastic, ATR-based volatility regime, 20/50/200 MA position, sector ETF relative strength, overnight gap size, day-of-week effect, month-of-year seasonality, earnings proximity filter
- For each suggestion: estimated accuracy, how to compute it, what threshold/condition to use

### 3. New Signals Requiring Paid Data (worth buying?)
Are there any paid data signals (options flow, dark pool, unusual activity, TICK/TRIN, Level 2) that have documented predictive power for 0DTE direction? If so, which single data feed would give the most edge per dollar?

### 4. Structural Improvements
Looking at the conviction band data and pair/triple combos — is there a smarter conviction formula or GO filter structure that would better separate winners from losers?

## Output Format
1. **Top 5 signal combinations** (pair or triple) worth implementing as a "super filter," with realistic expected win rate and annual trade count
2. **Top 5 new backtestable signals** with exact implementation guidance (yfinance-compatible Python pseudocode)
3. **Top 2 paid data signals** worth considering, with estimated cost and edge
4. **One structural change** to the conviction formula or GO logic if warranted
5. **Honest ceiling**: Given daily OHLCV data only, what is the realistic maximum win rate for this system?

Be specific and quantitative. Don't pad. If a combination only has 8 trades in 2 years, say so and note it's too thin.
"""

print("\n  Calling claude-fable-5-2... (30-120 seconds)\n")

client = anthropic.Anthropic()

with client.beta.messages.stream(
    model="claude-fable-5-2",
    max_tokens=16000,
    output_config={"effort": "high"},
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
    messages=[{"role": "user", "content": PROMPT}],
) as stream:
    response = stream.get_final_message()

result_text = ""
for block in response.content:
    if block.type == "text":
        result_text += block.text

print("=" * 72)
print(result_text)
print("=" * 72)

OUT = os.path.join(os.path.dirname(__file__), "fable_confluence_results.py")
with open(OUT, "w") as f:
    f.write("# Fable confluence analysis — generated by fable_confluence.py\n\n")
    f.write(result_text)

print(f"\n  Saved to {OUT}")
print(f"  Usage: {response.usage.input_tokens} in / {response.usage.output_tokens} out tokens")
