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

ENTRY_HOUR_CT   = 10   # 10:00 AM CT entry
ENTRY_MINUTE_CT = 0
OPEN_HOUR_CT    = 9    # market open 9:30 AM CT
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


def build_intraday_signals(ticker: str, bars: pd.DataFrame, day: date) -> dict | None:
    """
    Given a full day's minute bars for one ticker, compute:
      - entry_price:    first bar at or after 10:00 AM CT open price
      - close_price:    last bar at or before 3:00 PM CT close price
      - intraday_pct:   (close - entry) / entry * 100
      - opening_range_high / low: 9:30–10:00 AM CT range
      - vwap_at_entry:  VWAP from 9:30 AM to entry bar
      - entry_vs_vwap:  entry price vs VWAP (positive = above VWAP)
      - open_gap_pct:   (first bar open - prior day close) / prior day close * 100
                        (requires prior-day last bar)

    Returns None if data is insufficient.
    """
    day_bars = bars[bars.index.date == day]
    if len(day_bars) < 10:
        return None

    # Convert to CT for time-of-day filtering
    day_bars = day_bars.copy()
    day_bars["ct"] = day_bars.index.tz_convert(CT)

    entry_dt = datetime(day.year, day.month, day.day,
                        ENTRY_HOUR_CT, ENTRY_MINUTE_CT, tzinfo=CT)
    open_dt  = datetime(day.year, day.month, day.day,
                        OPEN_HOUR_CT, OPEN_MINUTE_CT, tzinfo=CT)
    close_dt = datetime(day.year, day.month, day.day,
                        CLOSE_HOUR_CT, CLOSE_MINUTE_CT, tzinfo=CT)

    # Entry bar
    entry_bars = day_bars[day_bars["ct"] >= entry_dt]
    if entry_bars.empty:
        return None
    entry_bar   = entry_bars.iloc[0]
    entry_price = float(entry_bar["open"])

    # Close bar
    close_bars = day_bars[day_bars["ct"] <= close_dt]
    if close_bars.empty:
        return None
    close_price = float(close_bars.iloc[-1]["close"])

    intraday_pct = round((close_price / entry_price - 1) * 100, 3)

    # Opening range: 9:30–10:00 AM CT
    or_bars = day_bars[(day_bars["ct"] >= open_dt) & (day_bars["ct"] < entry_dt)]
    or_high = float(or_bars["high"].max()) if not or_bars.empty else entry_price
    or_low  = float(or_bars["low"].min())  if not or_bars.empty else entry_price

    # VWAP from open to entry
    pre_entry = day_bars[day_bars["ct"] <= entry_dt]
    if not pre_entry.empty:
        typical = (pre_entry["high"] + pre_entry["low"] + pre_entry["close"]) / 3
        cum_vol  = pre_entry["volume"].cumsum()
        cum_tpv  = (typical * pre_entry["volume"]).cumsum()
        vwap = float(cum_tpv.iloc[-1] / cum_vol.iloc[-1]) if cum_vol.iloc[-1] > 0 else entry_price
    else:
        vwap = entry_price
    entry_vs_vwap = round((entry_price / vwap - 1) * 100, 3)

    return {
        "entry_price":      round(entry_price, 4),
        "close_price":      round(close_price, 4),
        "intraday_pct":     intraday_pct,
        "or_high":          round(or_high, 4),
        "or_low":           round(or_low, 4),
        "or_range_pct":     round((or_high / or_low - 1) * 100, 3) if or_low > 0 else 0,
        "vwap_at_entry":    round(vwap, 4),
        "entry_vs_vwap":    entry_vs_vwap,
    }


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
