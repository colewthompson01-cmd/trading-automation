#!/usr/bin/env python3
"""
backtest_0dte.py — Historical accuracy test for 0DTE scanner signals

Replays scanner signals against 2 years of daily data and measures:
  - Win rate by ticker, conviction level, and VIX regime
  - Which individual signals are most predictive
  - Signal accuracy breakdown for Fable weight optimization

Usage:
  python3 backtest_0dte.py                    # 2-year backtest, all tickers
  python3 backtest_0dte.py --years 1          # 1-year backtest
  python3 backtest_0dte.py --ticker SPY       # single ticker
  python3 backtest_0dte.py --min-conviction 6 # only score GO signals

Win condition:
  CALL + intraday return > WIN_THRESHOLD  → win
  PUT  + intraday return < -WIN_THRESHOLD → win
  (intraday = close vs open, proxy for 10 AM entry to close)

NOTE: News sentiment is set to NEUTRAL for all historical days (can't
      replay RSS). This is intentional — it shows you how much sentiment
      actually matters vs the other signals.
"""

import sys, os, argparse, json
from datetime import date, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import importlib.util

# ── CONFIG ────────────────────────────────────────────────────────────────
SCAN_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "META",
                "AMZN", "MSFT", "GOOGL", "AMD", "MU", "NFLX", "COIN",
                # Candidates under evaluation
                "MSTR", "PLTR", "CRWD", "ARM", "HOOD", "SQ", "PANW", "SMCI"]

SECTORS = {
    "Tech":        "XLK",
    "Financials":  "XLF",
    "Energy":      "XLE",
    "Healthcare":  "XLV",
    "Comms":       "XLC",
    "Industrials": "XLI",
}

CROSS_TICKERS = {
    "10Y Yield": "^TNX",
    "DXY":       "DX-Y.NYB",
    "Oil (WTI)": "CL=F",
    "Gold":      "GC=F",
    "Bitcoin":   "BTC-USD",
}

WIN_THRESHOLD = 0.3   # % intraday move to count as a win
YEARS_BACK    = 2

# ── IMPORT build_recommendation FROM SCANNER ─────────────────────────────
def _load_scanner():
    path = os.path.join(os.path.dirname(__file__), "0dte_scanner.py")
    spec = importlib.util.spec_from_file_location("scanner", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

print("Loading scanner module...", end=" ", flush=True)
_scanner = _load_scanner()
build_recommendation = _scanner.build_recommendation
print("OK")

# ── HELPERS ───────────────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return (100 - 100 / (1 + rs))

def calc_bb_pctb(series, period=20):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (series - lower) / (upper - lower)

def pct_chg(series):
    return series.pct_change() * 100

# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────
def download_all(years=YEARS_BACK):
    end   = date.today()
    start = end - timedelta(days=int(years * 365) + 60)  # extra buffer for indicators
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    all_syms = (SCAN_TICKERS +
                list(SECTORS.values()) +
                list(CROSS_TICKERS.values()) +
                ["^VIX", "ES=F"])

    print(f"Downloading {len(all_syms)} symbols from {start_str} to {end_str}...")
    raw = yf.download(all_syms, start=start_str, end=end_str,
                      auto_adjust=True, progress=True)

    # yf returns MultiIndex (field, ticker) when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"]
        open_  = raw["Open"]
    else:
        close  = raw[["Close"]].rename(columns={"Close": all_syms[0]})
        open_  = raw[["Open"]].rename(columns={"Open": all_syms[0]})

    return close, open_

# ── BUILD SIGNAL DICT FOR ONE DAY ────────────────────────────────────────
def build_day_signals(ticker, day_idx, close, open_, rsi_df, pctb_df):
    """
    Construct the tech/cross/sectors dicts for one historical trading day.
    Uses prior-day data only (no lookahead) — simulates what you'd know
    at market open before placing a 0DTE trade.
    """
    if day_idx < 2:
        return None, None, None

    prev_idx = day_idx - 1

    # ── Ticker technicals ────────────────────────────────────────────────
    if ticker not in close.columns:
        return None, None, None

    try:
        tk_close_prev = close[ticker].iloc[prev_idx]
        tk_open_today = open_[ticker].iloc[day_idx]
        tk_premkt     = round((tk_open_today / tk_close_prev - 1) * 100, 2)
        tk_rsi        = round(float(rsi_df[ticker].iloc[prev_idx]), 1)
        tk_pctb       = round(float(pctb_df[ticker].iloc[prev_idx]), 2)
    except Exception:
        return None, None, None

    # ── VIX ──────────────────────────────────────────────────────────────
    try:
        vix      = round(float(close["^VIX"].iloc[prev_idx]), 2)
        vix_prev = round(float(close["^VIX"].iloc[prev_idx - 1]), 2)
        vix_chg  = round(vix - vix_prev, 2)
    except Exception:
        vix = 20.0
        vix_chg = 0.0

    # ── Futures (ES overnight gap as proxy) ──────────────────────────────
    try:
        es_prev2 = float(close["ES=F"].iloc[prev_idx - 1])
        es_prev  = float(close["ES=F"].iloc[prev_idx])
        futures_pct = round((es_prev / es_prev2 - 1) * 100, 2)
    except Exception:
        futures_pct = tk_premkt  # fall back to ticker's own gap

    tech = {
        "tk_premkt":   tk_premkt,
        "tk_rsi":      tk_rsi,
        "tk_pctb":     tk_pctb,
        "vix":         vix,
        "vix_chg":     vix_chg,
        "futures_pct": futures_pct,
    }

    # ── Cross-asset ───────────────────────────────────────────────────────
    cross = {}
    for name, sym in CROSS_TICKERS.items():
        try:
            if sym in close.columns:
                p0 = float(close[sym].iloc[prev_idx - 1])
                p1 = float(close[sym].iloc[prev_idx])
                cross[name] = {"price": round(p1, 2),
                               "pct":   round((p1 / p0 - 1) * 100, 2)}
        except Exception:
            pass

    # ── Sectors ───────────────────────────────────────────────────────────
    sectors = {}
    for name, sym in SECTORS.items():
        try:
            if sym in close.columns:
                p0 = float(close[sym].iloc[prev_idx - 1])
                p1 = float(close[sym].iloc[prev_idx])
                sectors[name] = round((p1 / p0 - 1) * 100, 2)
        except Exception:
            pass

    return tech, cross, sectors

# ── RUN BACKTEST ──────────────────────────────────────────────────────────
def run_backtest(tickers, close, open_, years=YEARS_BACK):
    cutoff = date.today() - timedelta(days=int(years * 365))

    # Pre-compute RSI and BB%B for all tickers
    print("Computing RSI and BB%B indicators...")
    rsi_df  = close[tickers].apply(lambda s: calc_rsi(s, 14))
    pctb_df = close[tickers].apply(lambda s: calc_bb_pctb(s, 20))

    rows = []
    trading_days = [i for i, d in enumerate(close.index)
                    if d.date() >= cutoff]

    print(f"Backtesting {len(tickers)} tickers over {len(trading_days)} trading days...")

    for day_idx in trading_days:
        day_date = close.index[day_idx].date()

        for ticker in tickers:
            tech, cross, sectors = build_day_signals(
                ticker, day_idx, close, open_, rsi_df, pctb_df
            )
            if tech is None:
                continue

            # Sentiment = NEUTRAL (can't replay historical news)
            # cat_score baseline of 2 approximates an average trading day
            # (real scanner adds 0-5; without this, conviction never reaches 5)
            sentiment = "NEUTRAL"
            cat_score = 2

            try:
                direction, conviction, go_nogo, signals = build_recommendation(
                    tech, cross, sectors, sentiment, cat_score
                )
            except Exception:
                continue

            # Actual intraday return (open → close)
            try:
                day_open  = float(open_[ticker].iloc[day_idx])
                day_close = float(close[ticker].iloc[day_idx])
                intraday  = round((day_close / day_open - 1) * 100, 3)
            except Exception:
                continue

            # Win/loss determination
            if direction == "CALL":
                outcome = "WIN" if intraday >  WIN_THRESHOLD else \
                          ("LOSS" if intraday < -WIN_THRESHOLD else "FLAT")
            elif direction == "PUT":
                outcome = "WIN" if intraday < -WIN_THRESHOLD else \
                          ("LOSS" if intraday >  WIN_THRESHOLD else "FLAT")
            else:
                outcome = "NO_TRADE"

            # VIX regime
            vix = tech.get("vix", 20)
            if vix < 15:
                vix_regime = "LOW (<15)"
            elif vix > 28:
                vix_regime = "HIGH (>28)"
            else:
                vix_regime = "NORMAL (15-28)"

            # Conviction bucket
            if conviction >= 8:
                conv_bucket = "8-10"
            elif conviction >= 7:
                conv_bucket = "7-8"
            elif conviction >= 6:
                conv_bucket = "6-7"
            elif conviction >= 5:
                conv_bucket = "5-6"
            else:
                conv_bucket = "<5"

            # Individual signal votes
            sig_dict = {s[0]: s[2] for s in signals}

            rows.append({
                "date":        day_date,
                "ticker":      ticker,
                "direction":   direction,
                "conviction":  conviction,
                "conv_bucket": conv_bucket,
                "go_nogo":     go_nogo,
                "vix":         vix,
                "vix_regime":  vix_regime,
                "intraday":    intraday,
                "outcome":     outcome,
                **{f"sig_{k.replace(' ', '_')}": v
                   for k, v in sig_dict.items()},
            })

    return pd.DataFrame(rows)

# ── ANALYSIS ──────────────────────────────────────────────────────────────
def win_rate(df):
    """Win rate excluding NO_TRADE and FLAT days."""
    df2 = df[df["outcome"].isin(["WIN", "LOSS"])]
    if df2.empty:
        return 0.0, 0
    return round(df2["outcome"].eq("WIN").mean() * 100, 1), len(df2)

def print_section(title):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")

def analyze(df, min_conviction=0):
    go_df = df[
        df["direction"].isin(["CALL", "PUT"]) &
        (df["conviction"] >= min_conviction)
    ].copy()

    print("\n" + "="*62)
    print("  0DTE SCANNER BACKTEST RESULTS")
    print("="*62)
    print(f"  Date range:    {df['date'].min()}  →  {df['date'].max()}")
    print(f"  Trading days:  {df['date'].nunique()}")
    print(f"  Min conviction filter: {min_conviction}")

    # ── Overall ───────────────────────────────────────────────────────────
    print_section("OVERALL WIN RATE (all GO signals)")
    wr, n = win_rate(go_df)
    print(f"  {wr}%  ({n} decisive trades)")

    # ── By ticker ─────────────────────────────────────────────────────────
    print_section("WIN RATE BY TICKER")
    print(f"  {'Ticker':<8} {'Win%':>6}  {'Trades':>7}  {'Avg Conviction':>15}")
    active_tickers = go_df["ticker"].unique()
    for ticker in [t for t in SCAN_TICKERS if t in active_tickers]:
        t_df = go_df[go_df["ticker"] == ticker]
        wr, n = win_rate(t_df)
        avg_conv = round(t_df["conviction"].mean(), 1) if not t_df.empty else 0
        bar = "▓" * int(wr / 5)
        print(f"  {ticker:<8} {wr:>5}%  {n:>7}  {avg_conv:>14}  {bar}")

    # ── By conviction bucket ──────────────────────────────────────────────
    print_section("WIN RATE BY CONVICTION LEVEL")
    print(f"  {'Conviction':<12} {'Win%':>6}  {'Trades':>7}")
    for bucket in ["8-10", "7-8", "6-7", "5-6", "<5"]:
        b_df = go_df[go_df["conv_bucket"] == bucket]
        wr, n = win_rate(b_df)
        print(f"  {bucket:<12} {wr:>5}%  {n:>7}")

    # ── By VIX regime ─────────────────────────────────────────────────────
    print_section("WIN RATE BY VIX REGIME")
    print(f"  {'VIX Regime':<18} {'Win%':>6}  {'Trades':>7}")
    for regime in ["LOW (<15)", "NORMAL (15-28)", "HIGH (>28)"]:
        r_df = go_df[go_df["vix_regime"] == regime]
        wr, n = win_rate(r_df)
        print(f"  {regime:<18} {wr:>5}%  {n:>7}")

    # ── By direction ──────────────────────────────────────────────────────
    print_section("WIN RATE BY DIRECTION")
    for direction in ["CALL", "PUT"]:
        d_df = go_df[go_df["direction"] == direction]
        wr, n = win_rate(d_df)
        print(f"  {direction:<6}  {wr}%  ({n} trades)")

    # ── Individual signal accuracy ────────────────────────────────────────
    print_section("SIGNAL ACCURACY (how often each signal voted correctly)")
    print(f"  {'Signal':<26} {'Correct%':>9}  {'Votes':>7}  {'Note'}")
    sig_cols = [c for c in go_df.columns if c.startswith("sig_")]
    signal_stats = []
    for col in sig_cols:
        label = col.replace("sig_", "").replace("_", " ")
        sub   = go_df[go_df[col] != 0].copy()
        sub   = sub[sub["outcome"].isin(["WIN", "LOSS"])]
        if len(sub) < 20:
            continue
        # Signal voted correctly when: vote > 0 + WIN, or vote < 0 + LOSS
        # but direction is CALL/PUT, outcome is WIN/LOSS relative to that direction
        # vote == 1 means "supports direction"; vote == -1 means "opposes direction"
        correct = (
            ((sub[col] > 0) & (sub["outcome"] == "WIN")) |
            ((sub[col] < 0) & (sub["outcome"] == "LOSS"))
        ).mean() * 100
        signal_stats.append((label, round(correct, 1), len(sub)))

    signal_stats.sort(key=lambda x: x[1], reverse=True)
    for label, acc, n in signal_stats:
        note = "★ STRONG" if acc >= 55 else ("✗ weak" if acc < 48 else "")
        print(f"  {label:<26} {acc:>8}%  {n:>7}  {note}")

    # ── SPY specific deep dive ────────────────────────────────────────────
    print_section("SPY DEEP DIVE — Conviction ≥ 6 only")
    spy_df = go_df[(go_df["ticker"] == "SPY") & (go_df["conviction"] >= 6)]
    wr, n  = win_rate(spy_df)
    print(f"  Win rate: {wr}%  ({n} trades)")
    calls = spy_df[spy_df["direction"] == "CALL"]
    puts  = spy_df[spy_df["direction"] == "PUT"]
    cwr, cn = win_rate(calls)
    pwr, pn = win_rate(puts)
    print(f"  CALL: {cwr}% ({cn} trades)   PUT: {pwr}% ({pn} trades)")

    # ── Suggested weight adjustments (raw data for Fable) ─────────────────
    print_section("SIGNAL PERFORMANCE SUMMARY (copy this to Fable)")
    summary = {}
    for label, acc, n in signal_stats:
        summary[label] = {"accuracy_pct": acc, "sample_size": n}
    print(json.dumps(summary, indent=2))

    return go_df

# ── SAVE RESULTS ──────────────────────────────────────────────────────────
def save_results(df, path="backtest_results.csv"):
    df.to_csv(path, index=False)
    print(f"\n  Saved {len(df)} rows to {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="0DTE scanner backtest")
    parser.add_argument("--years",          type=float, default=YEARS_BACK)
    parser.add_argument("--ticker",         type=str,   default=None)
    parser.add_argument("--min-conviction", type=float, default=0)
    parser.add_argument("--save",           action="store_true")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else SCAN_TICKERS

    close, open_ = download_all(years=args.years)

    # Filter to tickers that exist in the downloaded data
    available = [t for t in tickers if t in close.columns]
    missing   = set(tickers) - set(available)
    if missing:
        print(f"  Warning: no data for {missing}")

    df = run_backtest(available, close, open_, years=args.years)

    if df.empty:
        print("No results — check date range or ticker list.")
        return

    results = analyze(df, min_conviction=args.min_conviction)

    out_path = os.path.join(os.path.dirname(__file__), "backtest_results.csv")
    save_results(results, out_path)
    print("\n  ★ To optimize weights: paste the SIGNAL PERFORMANCE SUMMARY above")
    print("    into Fable with your current signal_weights config and ask it")
    print("    to rewrite the weights to maximize win rate.\n")

if __name__ == "__main__":
    main()
