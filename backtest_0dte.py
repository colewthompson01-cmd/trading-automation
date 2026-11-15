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
from datetime import date, timedelta, datetime
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
import importlib.util

ALPACA_CACHE_DIR = Path(__file__).parent / "alpaca_cache"

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

# ── ALPACA DATA LOADER ────────────────────────────────────────────────────
def load_alpaca_data(tickers: list, start: date, end: date,
                     refresh: bool = False) -> dict:
    """
    Load Alpaca minute bars for all tickers, using a per-ticker parquet cache.
    Returns {ticker: DataFrame} with UTC-indexed minute bars.
    Falls back to empty dict if Alpaca keys are not set.
    """
    if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
        print("  ALPACA_API_KEY not set — skipping Alpaca data (run: source ~/.zshrc)")
        return {}

    _path = Path(__file__).parent / "alpaca_data.py"
    if not _path.exists():
        print("  alpaca_data.py not found — skipping")
        return {}

    spec = importlib.util.spec_from_file_location("alpaca_data", str(_path))
    amod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(amod)

    ALPACA_CACHE_DIR.mkdir(exist_ok=True)
    result = {}
    to_fetch = []

    for ticker in tickers:
        cache_file = ALPACA_CACHE_DIR / f"{ticker}.parquet"
        if not refresh and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df.index = pd.to_datetime(df.index, utc=True)
                cached_start = df.index.min().date()
                cached_end   = df.index.max().date()
                if cached_start <= start and cached_end >= (end - timedelta(days=3)):
                    result[ticker] = df
                    continue
            except Exception:
                pass
        to_fetch.append(ticker)

    if to_fetch:
        print(f"  Fetching Alpaca minute bars for {len(to_fetch)} tickers "
              f"({start} → {end})...")
        # Batch in groups of 10 to stay within rate limits
        for i in range(0, len(to_fetch), 10):
            batch = to_fetch[i:i+10]
            bars  = amod.fetch_intraday_batch(batch, start, end, verbose=False)
            for ticker, df in bars.items():
                if not df.empty:
                    cache_file = ALPACA_CACHE_DIR / f"{ticker}.parquet"
                    df.to_parquet(cache_file)
                    result[ticker] = df
                    print(f"    {ticker}: {len(df):,} bars cached")

    got = sum(1 for v in result.values() if not v.empty)
    print(f"  Alpaca: {got}/{len(tickers)} tickers loaded")

    # Pre-compute RVOL once per ticker — stored as _rvol_<ticker> key to avoid re-scanning
    print("  Pre-computing RVOL for each ticker...", end=" ", flush=True)
    for ticker, df in list(result.items()):
        if not df.empty:
            result[f"_rvol_{ticker}"] = amod.precompute_rvol(df)
    print("done")
    return result

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

    # Add ^VIX3M for VIX term structure signal; SMH for semi relative-strength
    extra_syms = ["^VIX3M", "SMH"]
    all_syms = (SCAN_TICKERS +
                list(SECTORS.values()) +
                list(CROSS_TICKERS.values()) +
                ["^VIX", "ES=F"] + extra_syms)

    print(f"Downloading {len(all_syms)} symbols from {start_str} to {end_str}...")
    raw = yf.download(all_syms, start=start_str, end=end_str,
                      auto_adjust=True, progress=True)

    # yf returns MultiIndex (field, ticker) when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"]
        open_  = raw["Open"]
        high_  = raw["High"]
        low_   = raw["Low"]
    else:
        close  = raw[["Close"]].rename(columns={"Close": all_syms[0]})
        open_  = raw[["Open"]].rename(columns={"Open": all_syms[0]})
        high_  = raw[["High"]].rename(columns={"High": all_syms[0]})
        low_   = raw[["Low"]].rename(columns={"Low": all_syms[0]})

    return close, open_, high_, low_

# ── BUILD SIGNAL DICT FOR ONE DAY ────────────────────────────────────────
def build_day_signals(ticker, day_idx, close, open_, high_, low_, rsi_df, pctb_df,
                      atr_df, ma20_df, ma50_df, alpaca_bars=None):
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

    # ── Gap ATR ───────────────────────────────────────────────────────────
    gap_atr = None
    try:
        atr_val = float(atr_df[ticker].iloc[prev_idx])
        if atr_val > 0:
            gap_atr = round((tk_open_today - tk_close_prev) / atr_val, 2)
    except Exception:
        pass

    # ── Trend regime (20/50 MA) ───────────────────────────────────────────
    trend_regime = None
    try:
        s20 = float(ma20_df[ticker].iloc[prev_idx])
        s50 = float(ma50_df[ticker].iloc[prev_idx])
        pc  = float(tk_close_prev)
        if pc > s20 and s20 > s50:
            trend_regime = 1
        elif pc < s20 and s20 < s50:
            trend_regime = -1
        else:
            trend_regime = 0
    except Exception:
        pass

    # ── Relative strength vs sector ETF (5-day) ──────────────────────────
    rel_strength = None
    try:
        etf_map = {"NVDA": "SMH", "AMD": "SMH", "MU": "SMH",
                   "TSLA": "XLY", "META": "XLC", "AMZN": "XLC",
                   "NFLX": "XLC", "MSTR": "XLF", "HOOD": "XLF"}
        etf_sym = etf_map.get(ticker)
        if etf_sym and etf_sym in close.columns and prev_idx >= 5:
            etf_rs = float(close[etf_sym].iloc[prev_idx]) / float(close[etf_sym].iloc[prev_idx - 5]) - 1
            tk_rs  = float(close[ticker].iloc[prev_idx])  / float(close[ticker].iloc[prev_idx - 5])  - 1
            rel_strength = round((tk_rs - etf_rs) * 100, 2)
    except Exception:
        pass

    # ── VIX term structure (VIX / VIX3M) ─────────────────────────────────
    vix_ts = None
    try:
        if "^VIX3M" in close.columns:
            vix3m = float(close["^VIX3M"].iloc[prev_idx])
            if vix3m > 0:
                vix_ts = round(vix / vix3m, 3)
    except Exception:
        pass

    tech = {
        "tk_premkt":    tk_premkt,
        "tk_rsi":       tk_rsi,
        "tk_pctb":      tk_pctb,
        "vix":          vix,
        "vix_chg":      vix_chg,
        "futures_pct":  futures_pct,
        "gap_atr":      gap_atr,
        "trend_regime": trend_regime,
        "rel_strength": rel_strength,
        "vix_ts":       vix_ts,
    }

    # ── Alpaca intraday signals (VWAP, ORB, RVOL, gap fill) ─────────────────
    alpaca_entry = None   # will replace yfinance open for P&L if available
    if alpaca_bars and ticker in alpaca_bars and not alpaca_bars[ticker].empty:
        try:
            day_date_val = close.index[day_idx].date()
            _path = Path(__file__).parent / "alpaca_data.py"
            spec  = importlib.util.spec_from_file_location("alpaca_data", str(_path))
            amod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(amod)
            _atr14      = float(atr_df[ticker].iloc[prev_idx]) if ticker in atr_df.columns else None
            _rvol_lookup = alpaca_bars.get(f"_rvol_{ticker}", {})
            sig = amod.build_intraday_signals(ticker, alpaca_bars[ticker], day_date_val,
                                              atr14=_atr14, rvol_lookup=_rvol_lookup)
            if sig:
                tech["entry_vs_vwap"] = sig["entry_vs_vwap"]
                tech["or_range_pct"]  = sig["or_range_pct"]
                tech["orb_direction"] = sig["orb_direction"]
                tech["rvol"]          = sig["rvol"]
                tech["or_atr"]        = sig["or_atr"]
                tech["gap_sig"]       = sig["gap_sig"]
                alpaca_entry = sig["entry_price"]
                tech["_alpaca_entry"] = alpaca_entry
                tech["_alpaca_close"] = sig["close_price"]
        except Exception:
            pass

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
def run_backtest(tickers, close, open_, high_, low_, years=YEARS_BACK, alpaca_bars=None):
    cutoff = date.today() - timedelta(days=int(years * 365))

    # Pre-compute RSI, BB%B, ATR(14), 20/50 MA for all tickers
    print("Computing RSI, BB%B, ATR, and MA indicators...")
    rsi_df  = close[tickers].apply(lambda s: calc_rsi(s, 14))
    pctb_df = close[tickers].apply(lambda s: calc_bb_pctb(s, 20))

    # ATR(14): true range rolling mean
    tk_high  = high_[[t for t in tickers if t in high_.columns]]
    tk_low   = low_[[t  for t in tickers if t in low_.columns]]
    tk_close = close[[t for t in tickers if t in close.columns]]
    tr = pd.concat([
        tk_high - tk_low,
        (tk_high - tk_close.shift(1)).abs(),
        (tk_low  - tk_close.shift(1)).abs(),
    ]).groupby(level=0).max()
    atr_df  = tr.rolling(14).mean()
    ma20_df = tk_close.rolling(20).mean()
    ma50_df = tk_close.rolling(50).mean()

    rows = []
    trading_days = [i for i, d in enumerate(close.index)
                    if d.date() >= cutoff]

    print(f"Backtesting {len(tickers)} tickers over {len(trading_days)} trading days...")

    for day_idx in trading_days:
        day_date = close.index[day_idx].date()

        for ticker in tickers:
            tech, cross, sectors = build_day_signals(
                ticker, day_idx, close, open_, high_, low_,
                rsi_df, pctb_df, atr_df, ma20_df, ma50_df,
                alpaca_bars=alpaca_bars,
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

            # Intraday return — use Alpaca 10AM→close if available, else open→close
            try:
                if tech.get("_alpaca_entry") and tech.get("_alpaca_close"):
                    day_open  = tech["_alpaca_entry"]
                    day_close = tech["_alpaca_close"]
                else:
                    day_open  = float(open_[ticker].iloc[day_idx])
                    day_close = float(close[ticker].iloc[day_idx])
                intraday = round((day_close / day_open - 1) * 100, 3)
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
    parser.add_argument("--alpaca",         action="store_true",
                        help="Use Alpaca minute bars for 10AM entry price + VWAP signal")
    parser.add_argument("--refresh-cache",  action="store_true",
                        help="Force re-download of Alpaca minute bar cache")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else SCAN_TICKERS

    close, open_, high_, low_ = download_all(years=args.years)

    # Filter to tickers that exist in the downloaded data
    available = [t for t in tickers if t in close.columns]
    missing   = set(tickers) - set(available)
    if missing:
        print(f"  Warning: no data for {missing}")

    # Optionally load Alpaca minute bars
    alpaca_bars = None
    if args.alpaca:
        end   = date.today()
        start = end - timedelta(days=int(args.years * 365) + 10)
        alpaca_bars = load_alpaca_data(available, start, end,
                                       refresh=args.refresh_cache)

    df = run_backtest(available, close, open_, high_, low_, years=args.years,
                      alpaca_bars=alpaca_bars)

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
