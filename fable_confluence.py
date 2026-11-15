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
    "scanner_tickers": ["NVDA", "TSLA", "META", "AMZN", "AMD", "MU", "NFLX", "MSTR", "HOOD",
                        "PLTR", "CRWD", "ARM", "HOOD", "PANW", "SMCI", "COIN", "MSTR"],
    "go_window": "conviction 7.0 to 8.0 (above 8.0 = over-signaled, 48.9% win rate; 7-8 band = 59.0%)",
    "data_source": "Alpaca 1-minute bars (free tier, 2yr history) + yfinance daily",
    "alpaca_upgrade": {
        "description": "Now have Alpaca 1-min bars for all tickers. Already computing at 10AM CT entry:",
        "already_computed": [
            "entry_price: true 10AM CT bar open (not daily open proxy)",
            "close_price: 3PM CT last bar close",
            "vwap_at_entry: VWAP from 9:30AM to 10AM",
            "entry_vs_vwap: entry price % above/below VWAP",
            "or_high / or_low: opening range 9:30-10:00AM high/low",
            "or_range_pct: opening range width as % of price",
        ],
        "not_yet_computed_candidates": [
            "orb_direction: is entry price above or below the opening range? (classic 0DTE signal)",
            "or_width_vs_atr: opening range width / 14-day ATR — narrow=coiled, wide=exhausted",
            "first_bar_momentum: direction of price drift in 9:30-10AM window",
            "volume_ratio: 9:30-10AM cumulative volume vs 30-day average for that window",
            "vwap_slope: is VWAP rising or falling at 10AM entry?",
            "gap_fill_check: did overnight gap fill before 10AM?",
            "or_midpoint_position: is entry above or below the OR midpoint?",
            "open_to_vwap_pct: how far has price moved from open to VWAP by 10AM?",
        ],
    },
    "confirmed_signal_accuracy_with_alpaca": {
        "10Y Yield": {"accuracy_pct": 57.8, "sample_size": 540},
        "VIX Direction": {"accuracy_pct": 57.6, "sample_size": 564},
        "S&P Futures": {"accuracy_pct": 55.9, "sample_size": 653},
        "VWAP Position": {"accuracy_pct": 55.8, "sample_size": 588},
        "Sector Pulse": {"accuracy_pct": 54.4, "sample_size": 665},
        "Pre-Market Move": {"accuracy_pct": 52.8, "sample_size": 618},
        "DXY": {"accuracy_pct": 52.4, "sample_size": 418},
    },
}

PROMPT = f"""You are a quantitative trading systems engineer specializing in 0DTE (zero days-to-expiry) options strategies.

I have a Python 0DTE scanner that generates GO/NO-GO signals. We just upgraded from yfinance daily data to Alpaca 1-minute bars, which improved our backtest from 50.9% to 55.1% win rate (681 trades, 2 years). The 7-8 conviction band hits 59.0%. I want to push further using the new intraday data.

## Current System
- Signals: VIX Direction (w=2.0), Sector Pulse (1.75), DXY (1.75), 10Y Yield (1.25), S&P Futures (1.0), Pre-Market Move (1.0), VWAP Position (1.0), RSI(14) (0.5), BB%B (0.5), News Sentiment (0.5), VIX IVR (0.75), Put/Call Ratio (0.75), Gap ATR (0.75), Trend Regime (1.0)
- GO only when: conviction in [7.0, 8.0] (formula: 10 × agreement × √participation + catalyst_adj)
- Scanner runs at 10:00 AM CT; we have Alpaca 1-min bars from 9:30 AM onward
- Structural gate: GO only if Pre-Market active AND (VIX Direction agrees OR both DXY+Futures agree)
- Yield veto: block when Pre-Market and 10Y Yield vote same direction (historically 47% win rate)

## 2-Year Backtest Data

```json
{json.dumps(DATA, indent=2)}
```

## Questions

### 1. New Intraday Signals (from Alpaca 1-min bars)
I now have per-minute OHLCV from 9:30-10:00 AM before my entry. From the candidate list in `alpaca_upgrade.not_yet_computed_candidates`, which signals have the strongest theoretical and empirical basis for 0DTE direction prediction?

For each you recommend:
- What threshold/condition signals CALL vs PUT vs NEUTRAL
- How to compute it from minute bars in Python
- Realistic expected accuracy range (be honest — 52% is still additive if consistent)
- Suggested weight in the conviction formula

### 2. Signal Combinations with Intraday Data
The best pair combos and triple combos are in the data. Now that we have VWAP Position confirmed at 55.8%, what new pairs/triples involving VWAP + intraday signals would you expect to perform best?

### 3. Revised Signal Weights
Given the confirmed accuracy data (10Y Yield 57.8%, VIX Direction 57.6%, S&P Futures 55.9%, VWAP Position 55.8%), should we rebalance weights? The current weights were set before VWAP was confirmed. Write out the full revised weight dict.

### 4. Veto/Gate Improvements
Are there any intraday conditions that should be hard vetoes (like our yield veto)? For example: OR range > 2× ATR (too volatile, skip), volume ratio < 0.5 (dead tape, skip)?

### 5. Honest ceiling
Given 1-minute Alpaca bars + yfinance daily, with no options flow or L2 data, what is the realistic maximum win rate ceiling? What single paid data feed would have the highest impact per dollar?

## Output Format
1. **Top 4 new intraday signals** to implement, with Python pseudocode for minute bars and conviction weight
2. **Revised signal weight dict** (full, copy-pasteable)
3. **New veto conditions** (if any) with exact threshold
4. **Honest ceiling** and best paid data recommendation

Be specific and quantitative. Don't pad. If a signal only adds 0.5% win rate, say so.
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
