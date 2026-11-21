#!/usr/bin/env python3
"""
backtest_orb.py — Historical ORB accuracy test for the orb_scanner watchlist

Uses 5-minute intraday data (last 60 days via yfinance) to replay
Opening Range Breakout setups and measure:
  - Win rate by ticker
  - Win rate by gap size bucket
  - Best entry time (first candle vs wait)
  - Best exit time (10:30 / 11:00 / 11:30 / 12:00 ET)
  - Option P&L approximation (ATM call/put with ~0.5 delta, 10:1 leverage)

Win condition (stock level):
  UP  gap + price rose from ORB entry to exit window  → win
  DOWN gap + price fell from ORB entry to exit window → win

Option approximation:
  Stock move % × delta (0.5) / (premium / stock_price) = option return %
  Capped at +150% gain, -100% loss (option expires worthless floor)

Usage:
  python3 backtest_orb.py                        # all tickers, 60-day window
  python3 backtest_orb.py --ticker NVDA          # single ticker
  python3 backtest_orb.py --min-gap 3.0          # only gaps >3%
  python3 backtest_orb.py --entry-candle 2       # wait for 2nd candle (9:40 ET)
"""

import argparse, sys, os
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ET = ZoneInfo('America/New_York')
ALPACA_CACHE = Path(__file__).parent / 'alpaca_cache'

# ── Config ────────────────────────────────────────────────────
WATCHLIST = [
    'NVDA','TSLA','AAPL','AMZN','MSFT','META',
    'GOOGL','AMD','NFLX','PLTR','CRWD','COIN','MSTR','ARM','HOOD',
]

MIN_GAP_PCT    = 2.0    # minimum gap to consider a setup
MAX_GAP_PCT    = 12.0   # skip gaps that are too extended
DELTA          = 0.50   # ATM option delta approximation
PREMIUM_PCT    = 0.015  # ATM 0DTE premium ≈ 1.5% of stock price (rough)
STOP_LOSS_OPT  = -0.50  # -50% option loss = stop
TARGET_OPT     = 1.00   # +100% option gain = target

# Exit windows in ET (hour, minute)
EXIT_WINDOWS = {
    '10:00 ET': (10, 0),
    '10:30 ET': (10, 30),
    '11:00 ET': (11, 0),
    '11:30 ET': (11, 30),
    '12:00 ET': (12, 0),
}

# Gap size buckets
GAP_BUCKETS = [
    (2.0, 3.0,  '2-3%'),
    (3.0, 5.0,  '3-5%'),
    (5.0, 8.0,  '5-8%'),
    (8.0, 12.0, '8-12%'),
]

# ── Data Fetchers ────────────────────────────────────────────

def fetch_via_alpaca(ticker, years):
    """
    Fetch 1-min bars via Alpaca, resample to 5-min.
    Caches to alpaca_cache/<ticker>_orb.parquet to avoid re-fetching.
    Returns 5-min DataFrame in ET, or None on failure.
    """
    try:
        from alpaca_data import fetch_minute_bars
    except ImportError:
        return None

    cache_file = ALPACA_CACHE / f'{ticker}_orb_{years}y.parquet'
    ALPACA_CACHE.mkdir(exist_ok=True)

    # Use cache if fresh (less than 12 hours old)
    if cache_file.exists():
        age_hrs = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
        if age_hrs < 12:
            try:
                df = pd.read_parquet(cache_file)
                print(f'  [{ticker}] loaded from cache ({len(df)} 5-min bars)')
                return df
            except Exception:
                pass

    end   = date.today()
    start = end - timedelta(days=years * 365 + 10)
    print(f'  [{ticker}] fetching {years}y Alpaca 1-min bars...')

    try:
        bars = fetch_minute_bars(ticker, start, end)
        if bars.empty:
            return None

        # Convert to ET
        bars.index = bars.index.tz_convert(ET)

        # Resample 1-min → 5-min
        df = bars.resample('5min', label='left', closed='left').agg(
            Open=('open', 'first'),
            High=('high', 'max'),
            Low=('low', 'min'),
            Close=('close', 'last'),
            Volume=('volume', 'sum'),
        ).dropna(subset=['Close'])
        df.columns = ['Open','High','Low','Close','Volume']

        df.to_parquet(cache_file)
        print(f'  [{ticker}] {len(df)} 5-min bars cached')
        return df
    except Exception as e:
        print(f'  [{ticker}] Alpaca error: {e}')
        return None


def fetch_via_yfinance(ticker):
    """Fetch 5-min intraday data for last 60 days via yfinance."""
    try:
        df = yf.Ticker(ticker).history(period='60d', interval='5m', prepost=False)
        if df.empty or len(df) < 50:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(ET)
        else:
            df.index = df.index.tz_localize(ET)
        df.columns = [c.capitalize() for c in df.columns]
        return df
    except Exception as e:
        print(f'  [{ticker}] yfinance error: {e}')
        return None


def fetch_intraday(ticker, use_alpaca, years):
    """Fetch 5-min data from Alpaca (if available) or yfinance."""
    if use_alpaca:
        df = fetch_via_alpaca(ticker, years)
        if df is not None:
            return df
        print(f'  [{ticker}] Alpaca failed — falling back to yfinance (60 days only)')
    return fetch_via_yfinance(ticker)


def fetch_daily(ticker, years):
    """Fetch daily OHLCV for prev-close calculation."""
    try:
        period = f'{years + 1}y' if years <= 4 else 'max'
        df = yf.Ticker(ticker).history(period=period, interval='1d')
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None

# ── Option Return Approximation ───────────────────────────────

def approx_option_return(stock_move_pct, direction):
    """
    Approximate option P&L from underlying % move.
    Uses delta=0.5, premium≈1.5% of stock price.
    Returns option return % (e.g. 0.80 = +80%).
    """
    # Stock move in direction of trade
    directional_move = stock_move_pct if direction == 'UP' else -stock_move_pct
    # Option move = stock_move × delta / premium_pct
    opt_return = (directional_move / 100 * DELTA) / PREMIUM_PCT
    # Cap at stop and target
    return max(STOP_LOSS_OPT, min(TARGET_OPT, opt_return))

# ── Backtest Single Ticker ────────────────────────────────────

def backtest_ticker(ticker, min_gap, max_gap, entry_candle, use_alpaca, years):
    """
    Returns list of trade dicts for one ticker.
    entry_candle: 1 = first 5-min candle (9:30-9:35), 2 = second (9:35-9:40)
    """
    intraday = fetch_intraday(ticker, use_alpaca, years)
    daily    = fetch_daily(ticker, years)

    if intraday is None or daily is None:
        print(f'  [{ticker}] SKIP — no data')
        return []

    trades = []
    trading_days = sorted(set(intraday.index.date))

    for day in trading_days:
        try:
            # Previous close
            day_dt     = pd.Timestamp(day)
            prev_days  = daily[daily.index < day_dt]
            if prev_days.empty:
                continue
            prev_close = float(prev_days['Close'].iloc[-1])

            # Today's 5-min bars
            day_bars = intraday[intraday.index.date == day].copy()
            if len(day_bars) < 6:
                continue

            # First bar must be 9:30 open
            first_bar = day_bars.iloc[0]
            if first_bar.name.hour != 9 or first_bar.name.minute != 30:
                continue

            open_price = float(first_bar['Open'])

            # Gap calculation vs previous close
            gap_pct   = (open_price - prev_close) / prev_close * 100
            abs_gap   = abs(gap_pct)
            direction = 'UP' if gap_pct > 0 else 'DOWN'

            if abs_gap < min_gap or abs_gap > max_gap:
                continue

            # ORB: use high/low of entry candle as breakout levels
            entry_bar = day_bars.iloc[entry_candle - 1]
            orb_high  = float(entry_bar['High'])
            orb_low   = float(entry_bar['Low'])

            # Next bar after entry candle = earliest entry
            if len(day_bars) <= entry_candle:
                continue
            breakout_bar = day_bars.iloc[entry_candle]

            # Determine if breakout confirmed in direction of gap
            if direction == 'UP':
                if float(breakout_bar['Open']) < orb_high * 0.998:
                    continue  # no breakout
                entry_price = float(breakout_bar['Open'])
            else:
                if float(breakout_bar['Open']) > orb_low * 1.002:
                    continue
                entry_price = float(breakout_bar['Open'])

            entry_time = breakout_bar.name

            # Gap bucket
            gap_bucket = 'other'
            for lo, hi, label in GAP_BUCKETS:
                if lo <= abs_gap < hi:
                    gap_bucket = label
                    break

            # Measure outcomes at each exit window
            exits = {}
            for window_name, (exit_h, exit_m) in EXIT_WINDOWS.items():
                exit_bars = day_bars[
                    (day_bars.index.hour > exit_h) |
                    ((day_bars.index.hour == exit_h) & (day_bars.index.minute >= exit_m))
                ]
                if exit_bars.empty:
                    # Market closed before exit window — use last bar
                    exit_price = float(day_bars['Close'].iloc[-1])
                else:
                    exit_price = float(exit_bars.iloc[0]['Open'])

                stock_move  = (exit_price - entry_price) / entry_price * 100
                opt_return  = approx_option_return(stock_move, direction)
                win_stock   = (stock_move > 0) if direction == 'UP' else (stock_move < 0)

                exits[window_name] = {
                    'exit_price':   round(exit_price, 2),
                    'stock_move':   round(stock_move, 2),
                    'opt_return':   round(opt_return * 100, 1),
                    'win':          win_stock,
                }

            trades.append({
                'ticker':       ticker,
                'date':         str(day),
                'prev_close':   round(prev_close, 2),
                'open_price':   round(open_price, 2),
                'entry_price':  round(entry_price, 2),
                'gap_pct':      round(gap_pct, 2),
                'abs_gap':      round(abs_gap, 2),
                'gap_bucket':   gap_bucket,
                'direction':    direction,
                'orb_high':     round(orb_high, 2),
                'orb_low':      round(orb_low, 2),
                **{f'exit_{k}_stock': v['stock_move'] for k, v in exits.items()},
                **{f'exit_{k}_opt':   v['opt_return']  for k, v in exits.items()},
                **{f'exit_{k}_win':   v['win']          for k, v in exits.items()},
            })

        except Exception:
            continue

    print(f'  [{ticker}] {len(trades)} ORB setups found')
    return trades

# ── Analysis ─────────────────────────────────────────────────

def analyze(df):
    if df.empty:
        print('\nNo trades to analyze.')
        return

    print('\n' + '=' * 62)
    print('  ORB BACKTEST RESULTS')
    print(f'  {len(df)} total setups · {df["ticker"].nunique()} tickers · 60-day window')
    print('=' * 62)

    exit_cols = [c for c in df.columns if c.startswith('exit_') and c.endswith('_win')]

    # ── 1. Overall win rate by exit window ───────────────────
    print('\n── WIN RATE BY EXIT TIME ──────────────────────────────')
    print(f'  {"Exit Window":<14} {"Win Rate":>9} {"Avg Stock Move":>15} {"Avg Opt Return":>15} {"N":>5}')
    print(f'  {"-"*14} {"-"*9} {"-"*15} {"-"*15} {"-"*5}')

    best_exit = None
    best_rate = 0
    for window in EXIT_WINDOWS:
        wcol     = f'exit_{window}_win'
        scol     = f'exit_{window}_stock'
        ocol     = f'exit_{window}_opt'
        if wcol not in df.columns:
            continue
        win_rate = df[wcol].mean() * 100
        avg_stk  = df[scol].mean()
        avg_opt  = df[ocol].mean()
        n        = len(df)
        marker   = ' ◄ BEST' if win_rate > best_rate else ''
        if win_rate > best_rate:
            best_rate = win_rate
            best_exit = window
        print(f'  {window:<14} {win_rate:>8.1f}% {avg_stk:>+14.2f}% {avg_opt:>+14.1f}% {n:>5}{marker}')

    # ── 2. Win rate by ticker ─────────────────────────────────
    print('\n── WIN RATE BY TICKER ─────────────────────────────────')
    best_col = f'exit_{best_exit}_win'
    opt_col  = f'exit_{best_exit}_opt'
    stk_col  = f'exit_{best_exit}_stock'

    ticker_stats = (
        df.groupby('ticker')
        .agg(
            setups    = (best_col, 'count'),
            win_rate  = (best_col, 'mean'),
            avg_opt   = (opt_col,  'mean'),
            avg_stock = (stk_col,  'mean'),
        )
        .sort_values('win_rate', ascending=False)
    )

    print(f'  {"Ticker":<8} {"Setups":>7} {"Win Rate":>9} {"Avg Opt%":>10} {"Avg Stk%":>10}')
    print(f'  {"-"*8} {"-"*7} {"-"*9} {"-"*10} {"-"*10}')
    for ticker, row in ticker_stats.iterrows():
        bar = '█' * int(row['win_rate'] * 10)
        print(f'  {ticker:<8} {int(row["setups"]):>7} {row["win_rate"]*100:>8.1f}% '
              f'{row["avg_opt"]:>+9.1f}% {row["avg_stock"]:>+9.2f}%  {bar}')

    # ── 3. Win rate by gap size ───────────────────────────────
    print('\n── WIN RATE BY GAP SIZE ───────────────────────────────')
    gap_stats = (
        df.groupby('gap_bucket')
        .agg(
            setups   = (best_col, 'count'),
            win_rate = (best_col, 'mean'),
            avg_opt  = (opt_col,  'mean'),
        )
        .sort_values('win_rate', ascending=False)
    )
    print(f'  {"Gap Bucket":<10} {"Setups":>7} {"Win Rate":>9} {"Avg Opt%":>10}')
    print(f'  {"-"*10} {"-"*7} {"-"*9} {"-"*10}')
    for bucket, row in gap_stats.iterrows():
        print(f'  {bucket:<10} {int(row["setups"]):>7} {row["win_rate"]*100:>8.1f}% {row["avg_opt"]:>+9.1f}%')

    # ── 4. Win rate by direction (UP gap vs DOWN gap) ─────────
    print('\n── WIN RATE BY GAP DIRECTION ──────────────────────────')
    dir_stats = (
        df.groupby('direction')
        .agg(
            setups   = (best_col, 'count'),
            win_rate = (best_col, 'mean'),
            avg_opt  = (opt_col,  'mean'),
        )
    )
    for direction, row in dir_stats.iterrows():
        print(f'  {direction:<6} {int(row["setups"]):>4} setups · '
              f'{row["win_rate"]*100:.1f}% win rate · {row["avg_opt"]:+.1f}% avg opt return')

    # ── 5. Summary ────────────────────────────────────────────
    overall_win = df[best_col].mean() * 100
    overall_opt = df[opt_col].mean()
    print(f'\n── SUMMARY ────────────────────────────────────────────')
    print(f'  Best exit window : {best_exit}')
    print(f'  Overall win rate : {overall_win:.1f}%')
    print(f'  Avg option return: {overall_opt:+.1f}% per trade')
    print(f'  Note: option returns are approximated (delta=0.5, premium=1.5% of stock)')
    print()

# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker',       default=None,  help='Single ticker to test')
    parser.add_argument('--min-gap',      type=float, default=MIN_GAP_PCT)
    parser.add_argument('--max-gap',      type=float, default=MAX_GAP_PCT)
    parser.add_argument('--entry-candle', type=int,   default=1,
                        help='1=first 5-min candle (9:30), 2=wait for 9:35 confirmation')
    parser.add_argument('--years',        type=int,   default=0,
                        help='Years of history via Alpaca (0 = yfinance 60-day fallback)')
    args = parser.parse_args()

    use_alpaca = args.years > 0
    years      = args.years if args.years > 0 else 0
    tickers    = [args.ticker] if args.ticker else WATCHLIST

    if use_alpaca and not (os.environ.get('ALPACA_API_KEY') and os.environ.get('ALPACA_SECRET_KEY')):
        print('⚠️  --years requires ALPACA_API_KEY and ALPACA_SECRET_KEY to be set.')
        print('   Run: source ~/.zshrc')
        sys.exit(1)

    window_label = f'{years}-year Alpaca window' if use_alpaca else '60-day yfinance window'
    print('=' * 62)
    print(f'  ORB BACKTEST — {window_label}')
    print(f'  Gap filter: {args.min_gap:.1f}%–{args.max_gap:.1f}% · '
          f'Entry: candle {args.entry_candle} (9:{25+args.entry_candle*5:02d} ET)')
    print('=' * 62 + '\n')

    all_trades = []
    for ticker in tickers:
        trades = backtest_ticker(ticker, args.min_gap, args.max_gap,
                                 args.entry_candle, use_alpaca, years)
        all_trades.extend(trades)

    if not all_trades:
        print('\nNo ORB setups found in this window.')
        return

    df = pd.DataFrame(all_trades)

    # Save CSV
    fname = f'backtest_orb_results.csv'
    df.to_csv(fname, index=False)
    print(f'\n[saved] {fname} ({len(df)} rows)')

    analyze(df)

if __name__ == '__main__':
    main()
