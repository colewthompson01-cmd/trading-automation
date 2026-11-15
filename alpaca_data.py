#!/usr/bin/env python3
"""
alpaca_data.py — Alpaca minute bar data module for 0DTE scanner backtest.

Replaces yfinance daily OHLCV with Alpaca 1-minute bars, enabling:
  - Accurate 10:00 AM CT entry price (not daily open proxy)
  - Opening range (9:30-10:00 AM) high/low as a signal
  - VWAP position at entry time
  - True intraday P&L (10:00 AM entry → close)

Usage:
    source ~/.zshrc   # loads ALPACA_API_KEY and ALPACA_SECRET_KEY
    python3 alpaca_data.py          # self-test
    python3 alpaca_data.py --ticker NVDA --days 5   # single ticker test
"""

import os, sys, argparse
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "alpaca-py"])
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

CT  = ZoneInfo("America/Chicago")
UTC = timezone.utc

ENTRY_HOUR_CT      = 10   # 10:00 AM CT = 11:00 AM ET — scanner entry
ENTRY_MINUTE_CT    = 0
# NYSE opens 9:30 AM ET = 8:30 AM CT. The true opening range is the first 30 min of trading.
TRUE_OPEN_HOUR_CT  = 8    # 8:30 AM CT = 9:30 AM ET — true market open
TRUE_OPEN_MINUTE_CT= 30
TRUE_OR_END_HOUR_CT= 9    # 9:00 AM CT = 10:00 AM ET — end of true 30-min OR
TRUE_OR_END_MINUTE_CT = 0
# Legacy: 9:30-10:00 AM CT window (kept for backward compat with existing or_range_pct)
OPEN_HOUR_CT    = 9
OPEN_MINUTE_CT  = 30
CLOSE_HOUR_CT   = 15  # market close 3:00 PM CT (4:00 ET)
CLOSE_MINUTE_CT = 0


def _client():
    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.\n"
            "Run: source ~/.zshrc"
        )
    return StockHistoricalDataClient(key, secret)


def fetch_minute_bars(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV bars for a ticker between start and end dates.
    Returns a DataFrame indexed by (symbol, timestamp) with UTC timestamps.
    """
    client = _client()
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=datetime.combine(start, datetime.min.time()).replace(tzinfo=UTC),
        end=datetime.combine(end,   datetime.min.time()).replace(tzinfo=UTC),
    )
    bars = client.get_stock_bars(req)
    df   = bars.df
    if df.empty:
        return df
    # Drop symbol level from MultiIndex — keep just timestamp
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def build_intraday_signals(ticker: str, bars: pd.DataFrame, day: date,
                           atr14: float = None,
                           rvol_lookup: dict = None) -> dict | None:
    """
    Given minute bars (multi-day OK) for one ticker, compute intraday signals for `day`.

    New signals (Fable v2):
      - orb_direction:   +1/0/-1 — price broke/held/broke the TRUE opening range
                         (9:30-10:00 AM ET = 8:30-9:00 AM CT) with 10% buffer + 5-bar hold
      - rvol:            9:30-10AM ET vol today / 20-session median for that window
      - or_atr:          true OR width / atr14 (None if atr14 not passed)
      - gap_sig:         +1 gap-and-go, -1 failed gap, 0 no gap / no fill signal

    Legacy signals kept unchanged (backward compat):
      - entry_price, close_price, intraday_pct
      - or_high/or_low/or_range_pct (9:30-10:00 AM CT window, legacy)
      - vwap_at_entry, entry_vs_vwap

    Returns None if data is insufficient.
    """
    day_bars = bars[bars.index.date == day]
    if len(day_bars) < 10:
        return None

    day_bars = day_bars.copy()
    day_bars["ct"] = day_bars.index.tz_convert(CT)

    entry_dt     = datetime(day.year, day.month, day.day, ENTRY_HOUR_CT,       ENTRY_MINUTE_CT,     tzinfo=CT)
    open_dt      = datetime(day.year, day.month, day.day, OPEN_HOUR_CT,        OPEN_MINUTE_CT,      tzinfo=CT)
    close_dt     = datetime(day.year, day.month, day.day, CLOSE_HOUR_CT,       CLOSE_MINUTE_CT,     tzinfo=CT)
    true_open_dt = datetime(day.year, day.month, day.day, TRUE_OPEN_HOUR_CT,   TRUE_OPEN_MINUTE_CT, tzinfo=CT)
    true_or_end  = datetime(day.year, day.month, day.day, TRUE_OR_END_HOUR_CT, TRUE_OR_END_MINUTE_CT, tzinfo=CT)

    # ── Entry and close bars ─────────────────────────────────────────────────
    entry_bars = day_bars[day_bars["ct"] >= entry_dt]
    if entry_bars.empty:
        return None
    entry_price = float(entry_bars.iloc[0]["open"])

    close_bars = day_bars[day_bars["ct"] <= close_dt]
    if close_bars.empty:
        return None
    close_price  = float(close_bars.iloc[-1]["close"])
    intraday_pct = round((close_price / entry_price - 1) * 100, 3)

    # ── Legacy opening range: 9:30-10:00 AM CT (window before scanner entry) ─
    or_bars = day_bars[(day_bars["ct"] >= open_dt) & (day_bars["ct"] < entry_dt)]
    or_high = float(or_bars["high"].max()) if not or_bars.empty else entry_price
    or_low  = float(or_bars["low"].min())  if not or_bars.empty else entry_price

    # ── VWAP from true market open to entry ──────────────────────────────────
    pre_entry = day_bars[(day_bars["ct"] >= true_open_dt) & (day_bars["ct"] <= entry_dt)]
    if not pre_entry.empty:
        typical  = (pre_entry["high"] + pre_entry["low"] + pre_entry["close"]) / 3
        cum_vol  = pre_entry["volume"].cumsum()
        cum_tpv  = (typical * pre_entry["volume"]).cumsum()
        vwap     = float(cum_tpv.iloc[-1] / cum_vol.iloc[-1]) if cum_vol.iloc[-1] > 0 else entry_price
    else:
        vwap = entry_price
    entry_vs_vwap = round((entry_price / vwap - 1) * 100, 3)

    # ── TRUE opening range: 8:30-9:00 AM CT (9:30-10:00 AM ET) ─────────────
    true_or_bars = day_bars[(day_bars["ct"] >= true_open_dt) & (day_bars["ct"] < true_or_end)]
    if not true_or_bars.empty:
        true_or_hi = float(true_or_bars["high"].max())
        true_or_lo = float(true_or_bars["low"].min())
    else:
        true_or_hi = true_or_lo = entry_price

    # ── ORB direction: break with 10% buffer sustained for 5 bars post-OR ───
    post_or = day_bars[(day_bars["ct"] >= true_or_end) & (day_bars["ct"] < entry_dt)]
    orb_dir = 0
    if not post_or.empty and true_or_hi > true_or_lo:
        buf    = 0.10 * (true_or_hi - true_or_lo)
        last5  = post_or["close"].tail(5)
        if entry_price > true_or_hi + buf and len(last5) >= 3 and (last5 > true_or_hi).all():
            orb_dir = 1
        elif entry_price < true_or_lo - buf and len(last5) >= 3 and (last5 < true_or_lo).all():
            orb_dir = -1

    # ── Relative volume: today's OR window vs 20-session median ─────────────
    today_vol = true_or_bars["volume"].sum() if not true_or_bars.empty else 0
    # rvol_lookup: pre-computed {date: rvol} dict (pass from precompute_rvol for speed)
    # Falls back to None if not provided — scanner passes it, backtest pre-computes
    rvol = rvol_lookup.get(day) if rvol_lookup else None

    # ── OR width / ATR ───────────────────────────────────────────────────────
    or_atr = None
    if atr14 and atr14 > 0 and true_or_hi > true_or_lo:
        or_atr = round((true_or_hi - true_or_lo) / atr14, 3)

    # ── Gap fill state ───────────────────────────────────────────────────────
    gap_sig    = 0
    prev_close = None
    if day_idx >= 1:
        prev_day      = all_days_sorted[day_idx - 1]
        prev_day_bars = bars[bars.index.date == prev_day].copy()
        prev_day_bars["ct"] = prev_day_bars.index.tz_convert(CT)
        prev_close_bars = prev_day_bars[prev_day_bars["ct"] <= datetime(prev_day.year, prev_day.month, prev_day.day, CLOSE_HOUR_CT, CLOSE_MINUTE_CT, tzinfo=CT)]
        if not prev_close_bars.empty:
            prev_close = float(prev_close_bars.iloc[-1]["close"])

    if prev_close is not None and true_or_hi > true_or_lo:
        gap   = entry_price - prev_close
        gap_n = abs(gap) / (true_or_hi - true_or_lo)
        if abs(gap_n) >= 0.25:
            s       = int(np.sign(gap))
            day_lo  = float(day_bars["low"].min())
            day_hi  = float(day_bars["high"].max())
            filled  = (day_lo <= prev_close) if s > 0 else (day_hi >= prev_close)
            if not filled:
                gap_sig = s      # gap-and-go: trade in gap direction
            elif filled and int(np.sign(entry_price - prev_close)) == -s:
                gap_sig = -s     # failed gap: fade the gap

    return {
        "entry_price":      round(entry_price, 4),
        "close_price":      round(close_price, 4),
        "intraday_pct":     intraday_pct,
        # legacy OR (9:30-10:00 AM CT)
        "or_high":          round(or_high, 4),
        "or_low":           round(or_low, 4),
        "or_range_pct":     round((or_high / or_low - 1) * 100, 3) if or_low > 0 else 0,
        # VWAP
        "vwap_at_entry":    round(vwap, 4),
        "entry_vs_vwap":    entry_vs_vwap,
        # new Fable v2 signals
        "true_or_high":     round(true_or_hi, 4),
        "true_or_low":      round(true_or_lo, 4),
        "orb_direction":    orb_dir,
        "rvol":             rvol,
        "or_atr":           or_atr,
        "gap_sig":          gap_sig,
        "prev_close":       round(prev_close, 4) if prev_close else None,
    }


def precompute_rvol(bars: pd.DataFrame, window: int = 20) -> dict:
    """
    Pre-compute RVOL for every trading day. Returns {date: rvol_float}.
    Uses pandas-native ops only — no Python-level date loops.
    """
    if bars.empty:
        return {}

    ct_idx    = bars.index.tz_convert(CT)
    ct_hour   = ct_idx.hour
    ct_minute = ct_idx.minute
    # normalize() gives midnight Timestamp in CT — avoids slow .date property
    ct_day    = ct_idx.normalize()

    or_mask = (
        ((ct_hour == TRUE_OPEN_HOUR_CT)  & (ct_minute >= TRUE_OPEN_MINUTE_CT)) |
        ((ct_hour == TRUE_OR_END_HOUR_CT) & (ct_minute < TRUE_OR_END_MINUTE_CT))
    )

    vol_series         = pd.Series(bars["volume"].values, index=ct_day, dtype="float64")
    or_vol_by_day      = vol_series[or_mask].groupby(level=0).sum()
    rolling_median     = or_vol_by_day.shift(1).rolling(window, min_periods=5).median()
    rvol_series        = (or_vol_by_day / rolling_median).round(2)

    return {ts.date(): (float(v) if pd.notna(v) else None)
            for ts, v in rvol_series.items()}


def fetch_intraday_batch(tickers: list[str], start: date, end: date,
                         verbose: bool = True) -> dict[str, pd.DataFrame]:
    """
    Fetch minute bars for multiple tickers at once (single API call per ticker).
    Returns {ticker: DataFrame} where DataFrame is indexed by UTC timestamp.
    """
    client = _client()
    if verbose:
        print(f"  Fetching Alpaca minute bars: {len(tickers)} tickers "
              f"{start} → {end}...")

    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Minute,
        start=datetime.combine(start, datetime.min.time()).replace(tzinfo=UTC),
        end=datetime.combine(end,   datetime.min.time()).replace(tzinfo=UTC),
    )
    bars = client.get_stock_bars(req)
    df   = bars.df

    if df.empty:
        return {t: pd.DataFrame() for t in tickers}

    result = {}
    for ticker in tickers:
        try:
            tk_df = df.xs(ticker, level="symbol").copy()
            tk_df.index = pd.to_datetime(tk_df.index, utc=True)
            result[ticker] = tk_df
        except KeyError:
            result[ticker] = pd.DataFrame()

    if verbose:
        got = sum(1 for v in result.values() if not v.empty)
        print(f"  Got data for {got}/{len(tickers)} tickers")
    return result


def get_entry_price(ticker: str, day: date, bars: pd.DataFrame = None) -> float | None:
    """
    Get the 10:00 AM CT open price for a ticker on a given day.
    Pass bars= if you already have the minute data to avoid a re-fetch.
    """
    if bars is None:
        bars = fetch_minute_bars(ticker, day, day + timedelta(days=1))
    sig = build_intraday_signals(ticker, bars, day)
    return sig["entry_price"] if sig else None


# ── SELF-TEST ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--days",   type=int, default=3)
    args = parser.parse_args()

    end   = date.today()
    start = end - timedelta(days=args.days + 2)

    print(f"\nFetching {args.ticker} minute bars {start} → {end}...")
    bars = fetch_minute_bars(args.ticker, start, end)
    print(f"  Total bars: {len(bars)}")

    trading_days = sorted(set(bars.index.date))[-args.days:]
    print(f"\n  Intraday signals for last {len(trading_days)} trading day(s):\n")

    for day in trading_days:
        sig = build_intraday_signals(args.ticker, bars, day)
        if sig:
            print(f"  {day}")
            print(f"    Entry (10AM CT):  ${sig['entry_price']:.2f}")
            print(f"    Close:            ${sig['close_price']:.2f}")
            print(f"    Intraday return:  {sig['intraday_pct']:+.2f}%")
            print(f"    Opening range:    ${sig['or_low']:.2f} – ${sig['or_high']:.2f}  "
                  f"({sig['or_range_pct']:.2f}% wide)")
            print(f"    VWAP at entry:    ${sig['vwap_at_entry']:.2f}  "
                  f"(entry {sig['entry_vs_vwap']:+.2f}% vs VWAP)")
        else:
            print(f"  {day}  — insufficient data")
