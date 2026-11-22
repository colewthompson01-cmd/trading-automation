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
    'NVDA','TSLA','AAPL','MSFT','META',
    'AMD','NFLX','COIN','MSTR','ARM','HOOD',
]

UP_GAPS_ONLY   = True   # filter 2: only trade UP gaps (57% win vs 44% DOWN)
MIN_GAP_PCT    = 3.0    # filter 3: minimum gap — drop noisy 2-3% bucket
MAX_GAP_PCT    = 12.0   # skip gaps that are too extended
EARNINGS_BUFFER = 3     # filter 4: skip within N trading days of earnings
EXCLUDE_LOW_VIX = True  # filter 6: skip setups when prior-day VIX < 15

# filter 5: sector ETF alignment — ticker → ETF to check
SECTOR_ETF = {
    'AMD': 'SOXX', 'ARM': 'SOXX', 'NVDA': 'SOXX',
    'COIN': 'BITQ', 'MSTR': 'BITQ',
    'META': 'QQQ', 'MSFT': 'QQQ', 'AAPL': 'QQQ',
    'TSLA': 'QQQ', 'NFLX': 'QQQ', 'HOOD': 'QQQ', 'PLTR': 'QQQ',
}
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
    (3.0, 5.0,  '3-5%'),
    (5.0, 8.0,  '5-8%'),
    (8.0, 12.0, '8-12%'),
]

# ── Data Fetchers ────────────────────────────────────────────

def fetch_via_polygon(ticker, years):
    """
    Fetch 1-min bars via Polygon.io, resample to 5-min.
    Caches to alpaca_cache/<ticker>_orb_Ny_polygon.parquet.
    Free tier: 5 req/min — adds small delays between pages.
    Returns 5-min DataFrame in ET, or None on failure.
    """
    import requests, time as _time

    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        return None

    cache_file = ALPACA_CACHE / f'{ticker}_orb_{years}y_polygon.parquet'
    ALPACA_CACHE.mkdir(exist_ok=True)

    if cache_file.exists():
        age_hrs = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
        if age_hrs < 168:  # 7 days
            try:
                df = pd.read_parquet(cache_file)
                print(f'  [{ticker}] loaded from cache ({len(df)} 5-min bars)')
                return df
            except Exception:
                pass

    end   = date.today()
    start = end - timedelta(days=years * 365 + 10)
    print(f'  [{ticker}] fetching {years}y Polygon 1-min bars...')

    all_bars = []
    url = (
        f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/'
        f'{start}/{end}'
        f'?adjusted=true&sort=asc&limit=50000&apiKey={api_key}'
    )

    while url:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                _time.sleep(12)
                continue
            if r.status_code != 200:
                print(f'  [{ticker}] Polygon error: {r.status_code} {r.text[:120]}')
                return None
            data = r.json()
            all_bars.extend(data.get('results') or [])
            next_url = data.get('next_url')
            url = (next_url + f'&apiKey={api_key}') if next_url else None
            if url:
                _time.sleep(0.5)   # stay under 5 req/min on free tier
        except Exception as e:
            print(f'  [{ticker}] Polygon fetch error: {e}')
            return None

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars)
    df.index = pd.to_datetime(df['t'], unit='ms', utc=True).dt.tz_convert(ET)
    df = df.rename(columns={'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'})
    df = df[['Open','High','Low','Close','Volume']]

    # Resample 1-min → 5-min
    df = df.resample('5min', label='left', closed='left').agg(
        Open=('Open', 'first'),
        High=('High', 'max'),
        Low=('Low', 'min'),
        Close=('Close', 'last'),
        Volume=('Volume', 'sum'),
    ).dropna(subset=['Close'])

    df.to_parquet(cache_file)
    print(f'  [{ticker}] {len(df)} 5-min bars cached')
    return df


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


def fetch_intraday(ticker, use_polygon, years):
    """Fetch 5-min data from Polygon (if --years given) or yfinance."""
    if use_polygon:
        df = fetch_via_polygon(ticker, years)
        if df is not None:
            return df
        print(f'  [{ticker}] Polygon failed — falling back to yfinance (60 days only)')
    return fetch_via_yfinance(ticker)


def fetch_daily(ticker, years):
    """Fetch daily OHLCV for prev-close + SMA20 calculation."""
    try:
        period = f'{years + 1}y' if years <= 4 else 'max'
        df = yf.Ticker(ticker).history(period=period, interval='1d')
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df['SMA20'] = df['Close'].rolling(20, min_periods=10).mean()
        return df
    except Exception:
        return None


def fetch_vix(years):
    """Fetch VIX daily closes. Returns Series indexed by Timestamp."""
    try:
        period = f'{years + 1}y' if years <= 4 else 'max'
        df = yf.Ticker('^VIX').history(period=period, interval='1d')
        if df.empty:
            return pd.Series(dtype=float)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df['Close'].sort_index()
    except Exception:
        return pd.Series(dtype=float)

# ── Earnings Dates ───────────────────────────────────────────

def fetch_earnings_dates(ticker, years):
    """Return a set of dates within EARNINGS_BUFFER trading days of any earnings."""
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return set()
        dates = pd.to_datetime(ed.index).normalize().tz_localize(None)
        # Build set of all dates within buffer window around each earnings date
        blocked = set()
        all_trading = pd.bdate_range(
            end=date.today(), periods=(years + 1) * 365
        )
        all_trading_dates = set(all_trading.date)
        for ed_date in dates.date:
            # Find position in trading calendar
            nearby = [d for d in all_trading_dates
                      if abs((pd.Timestamp(d) - pd.Timestamp(ed_date)).days) <= EARNINGS_BUFFER * 2]
            # Count trading days distance
            sorted_cal = sorted(all_trading_dates)
            if ed_date in sorted_cal:
                idx = sorted_cal.index(ed_date)
                lo = max(0, idx - EARNINGS_BUFFER)
                hi = min(len(sorted_cal) - 1, idx + EARNINGS_BUFFER)
                for i in range(lo, hi + 1):
                    blocked.add(sorted_cal[i])
        return blocked
    except Exception:
        return set()


# ── Sector ETF Daily Gaps ─────────────────────────────────────

_etf_gap_cache = {}

def get_etf_gap_lookup(etf, years):
    """Return dict {date: gap_pct} for the given ETF."""
    if etf in _etf_gap_cache:
        return _etf_gap_cache[etf]
    try:
        daily = fetch_daily(etf, years)
        if daily is None or daily.empty:
            _etf_gap_cache[etf] = {}
            return {}
        gaps = {}
        prev_close = None
        for dt, row in daily.iterrows():
            d = dt.date() if hasattr(dt, 'date') else dt
            if prev_close is not None:
                gap = (float(row['Open']) - prev_close) / prev_close * 100
                gaps[d] = gap
            prev_close = float(row['Close'])
        _etf_gap_cache[etf] = gaps
        return gaps
    except Exception:
        _etf_gap_cache[etf] = {}
        return {}


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

def backtest_ticker(ticker, min_gap, max_gap, entry_candle, use_polygon, years, vix_series):
    """
    Returns list of trade dicts for one ticker.
    entry_candle: 1 = first 5-min candle (9:30-9:35), 2 = second (9:35-9:40)
    Each trade includes 5 filter columns: trend_aligned, rvol_ok, pm_vol_ok,
    is_monday, vix_regime.
    """
    intraday = fetch_intraday(ticker, use_polygon, years)
    daily    = fetch_daily(ticker, years)

    if intraday is None or daily is None:
        print(f'  [{ticker}] SKIP — no data')
        return []

    # ── Filter 4: earnings proximity ─────────────────────────────────────────
    earnings_blocked = fetch_earnings_dates(ticker, years)

    # ── Filter 5: sector ETF alignment ───────────────────────────────────────
    etf_sym    = SECTOR_ETF.get(ticker)
    etf_gaps   = get_etf_gap_lookup(etf_sym, years) if etf_sym else {}

    # ── Pre-compute RVOL: rolling 20-day avg of 9:30 candle volume ──────────
    open_slot = intraday[(intraday.index.hour == 9) & (intraday.index.minute == 30)]
    open_vols = open_slot['Volume'].groupby(open_slot.index.date).first()
    rolling_avg_vol = open_vols.rolling(20, min_periods=5).mean().shift(1)
    rvol_lookup = (open_vols / rolling_avg_vol).to_dict()   # {date: rvol_ratio}

    # ── Pre-compute pre-market volume ratio vs 20-day avg daily volume ───────
    pm_data = intraday.between_time('04:00', '09:29')
    pm_vols_by_day = pm_data.groupby(pm_data.index.date)['Volume'].sum()
    # 20-day rolling avg daily volume from daily data
    avg_dvol = daily['Volume'].rolling(20, min_periods=5).mean().shift(1)
    avg_dvol.index = avg_dvol.index.normalize()
    pm_ratio_lookup = {}
    for d, pm_vol in pm_vols_by_day.items():
        d_ts   = pd.Timestamp(d)
        past   = avg_dvol[avg_dvol.index < d_ts]
        avg_dv = float(past.iloc[-1]) if not past.empty and not pd.isna(past.iloc[-1]) else None
        pm_ratio_lookup[d] = (pm_vol / avg_dv) if avg_dv else None

    trades = []
    trading_days = sorted(set(intraday.index.date))

    for day in trading_days:
        try:
            day_dt    = pd.Timestamp(day)
            prev_days = daily[daily.index < day_dt]
            if prev_days.empty:
                continue
            prev_row   = prev_days.iloc[-1]
            prev_close = float(prev_row['Close'])

            # Today's 5-min bars — market hours only (9:30–16:00 ET)
            day_bars = intraday[intraday.index.date == day].copy()
            day_bars = day_bars.between_time('09:30', '15:55')
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

            if UP_GAPS_ONLY and direction != 'UP':
                continue

            # ORB: use high/low of entry candle as breakout levels
            entry_bar = day_bars.iloc[entry_candle - 1]
            orb_high  = float(entry_bar['High'])
            orb_low   = float(entry_bar['Low'])

            if len(day_bars) <= entry_candle:
                continue
            breakout_bar = day_bars.iloc[entry_candle]

            # Breakout confirmation in direction of gap
            if direction == 'UP':
                if float(breakout_bar['Open']) < orb_high * 0.998:
                    continue
                entry_price = float(breakout_bar['Open'])
            else:
                if float(breakout_bar['Open']) > orb_low * 1.002:
                    continue
                entry_price = float(breakout_bar['Open'])

            # Gap bucket
            gap_bucket = 'other'
            for lo, hi, label in GAP_BUCKETS:
                if lo <= abs_gap < hi:
                    gap_bucket = label
                    break

            # ── Filter signals ───────────────────────────────────────────────

            # 1. Monday flag
            is_monday = (day.weekday() == 0)

            # 2. Trend alignment: direction matches stock vs SMA20
            prev_sma = float(prev_row['SMA20']) if not pd.isna(prev_row['SMA20']) else None
            if prev_sma is not None:
                above_sma     = prev_close > prev_sma
                trend_aligned = (direction == 'UP' and above_sma) or \
                                (direction == 'DOWN' and not above_sma)
            else:
                trend_aligned = None

            # 3. RVOL >= 2x
            rvol     = rvol_lookup.get(day)
            rvol_ok  = bool(rvol is not None and rvol >= 2.0)

            # 4. Pre-market volume >= 15% of avg daily volume
            pm_ratio = pm_ratio_lookup.get(day)
            pm_vol_ok = bool(pm_ratio is not None and pm_ratio >= 0.15)

            # 5. VIX regime
            past_vix = vix_series[vix_series.index < day_dt] if not vix_series.empty else pd.Series()
            if not past_vix.empty:
                vix_val    = float(past_vix.iloc[-1])
                vix_regime = 'low' if vix_val < 15 else ('medium' if vix_val < 25 else 'high')
            else:
                vix_val    = None
                vix_regime = 'unknown'

            # 6. VIX >= 15 guard (filter 6)
            vix_ok = (vix_regime in ('medium', 'high'))

            # 7. Earnings proximity (filter 4)
            near_earnings = (day in earnings_blocked)

            # 8. Sector ETF alignment (filter 5)
            etf_gap = etf_gaps.get(day)
            etf_aligned = bool(etf_gap is not None and etf_gap > 0) if direction == 'UP' \
                          else bool(etf_gap is not None and etf_gap < 0)

            # Measure outcomes at each exit window
            exits = {}
            for window_name, (exit_h, exit_m) in EXIT_WINDOWS.items():
                exit_bars = day_bars[
                    (day_bars.index.hour > exit_h) |
                    ((day_bars.index.hour == exit_h) & (day_bars.index.minute >= exit_m))
                ]
                exit_price = float(exit_bars.iloc[0]['Open']) if not exit_bars.empty \
                             else float(day_bars['Close'].iloc[-1])
                stock_move = (exit_price - entry_price) / entry_price * 100
                opt_return = approx_option_return(stock_move, direction)
                win_stock  = (stock_move > 0) if direction == 'UP' else (stock_move < 0)
                exits[window_name] = {
                    'exit_price': round(exit_price, 2),
                    'stock_move': round(stock_move, 2),
                    'opt_return': round(opt_return * 100, 1),
                    'win':        win_stock,
                }

            trades.append({
                'ticker':        ticker,
                'date':          str(day),
                'prev_close':    round(prev_close, 2),
                'open_price':    round(open_price, 2),
                'entry_price':   round(entry_price, 2),
                'gap_pct':       round(gap_pct, 2),
                'abs_gap':       round(abs_gap, 2),
                'gap_bucket':    gap_bucket,
                'direction':     direction,
                'orb_high':      round(orb_high, 2),
                'orb_low':       round(orb_low, 2),
                # filter columns
                'is_monday':     is_monday,
                'trend_aligned': trend_aligned,
                'rvol':          round(rvol, 2) if rvol is not None else None,
                'rvol_ok':       rvol_ok,
                'pm_vol_ratio':  round(pm_ratio, 3) if pm_ratio is not None else None,
                'pm_vol_ok':     pm_vol_ok,
                'vix':           round(vix_val, 1) if vix_val is not None else None,
                'vix_regime':    vix_regime,
                'vix_ok':        vix_ok,
                'near_earnings': near_earnings,
                'etf_sym':       etf_sym,
                'etf_gap':       round(etf_gap, 2) if etf_gap is not None else None,
                'etf_aligned':   etf_aligned,
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
    print(f'  {len(df)} total setups · {df["ticker"].nunique()} tickers')
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

    # ── 5. Filter impact ─────────────────────────────────────
    base_win = df[best_col].mean() * 100
    base_opt = df[opt_col].mean()
    n_total  = len(df)

    filter_defs = [
        ('Trend aligned',     df.get('trend_aligned') == True),
        ('RVOL >= 2x',        df.get('rvol_ok') == True),
        ('Pre-mkt vol ≥15%',  df.get('pm_vol_ok') == True),
        ('No Monday',         df.get('is_monday') == False),
        ('VIX >= 15',         df.get('vix_ok', pd.Series(dtype=bool)) == True),
        ('No earnings ±3d',   df.get('near_earnings', pd.Series(dtype=bool)) == False),
        ('Sector ETF aligned',df.get('etf_aligned', pd.Series(dtype=bool)) == True),
    ]

    print(f'\n── FILTER IMPACT (exit: {best_exit}) ──────────────────────')
    print(f'  {"Filter":<24} {"N":>5} {"Pct":>5} {"Win%":>7} {"Opt%":>8} {"vs Base":>8}')
    print(f'  {"-"*24} {"-"*5} {"-"*5} {"-"*7} {"-"*8} {"-"*8}')
    print(f'  {"Baseline (all)":<24} {n_total:>5} {"100%":>5} {base_win:>6.1f}% {base_opt:>+7.1f}% {"—":>8}')

    combo_mask = pd.Series([True] * n_total, index=df.index)
    for name, mask in filter_defs:
        sub  = df[mask]
        n    = len(sub)
        if n < 5:
            print(f'  {name:<24} {n:>5} {"":>5} {"n/a":>7} {"n/a":>8} {"":>8}')
            continue
        win  = sub[best_col].mean() * 100
        opt  = sub[opt_col].mean()
        pct  = n / n_total * 100
        diff = win - base_win
        print(f'  {name:<24} {n:>5} {pct:>4.0f}% {win:>6.1f}% {opt:>+7.1f}% {diff:>+7.1f}pp')
        combo_mask &= mask

    sub_combo = df[combo_mask]
    n_combo = len(sub_combo)
    label = f'All {len(filter_defs)} filters'
    if n_combo >= 3:
        win  = sub_combo[best_col].mean() * 100
        opt  = sub_combo[opt_col].mean()
        pct  = n_combo / n_total * 100
        diff = win - base_win
        print(f'  {label:<24} {n_combo:>5} {pct:>4.0f}% {win:>6.1f}% {opt:>+7.1f}% {diff:>+7.1f}pp')
    else:
        print(f'  {label:<24} {n_combo:>5} {"":>5} {"n/a":>7} {"n/a":>8} {"too few":>8}')

    # ── 6. VIX regime breakdown ───────────────────────────────
    if 'vix_regime' in df.columns and df['vix_regime'].ne('unknown').any():
        print(f'\n── WIN RATE BY VIX REGIME ─────────────────────────────')
        regime_labels = {'low': 'Low  (<15)', 'medium': 'Med  (15-25)', 'high': 'High (>25)'}
        for regime in ['low', 'medium', 'high']:
            sub = df[df['vix_regime'] == regime]
            if len(sub) < 3:
                continue
            win = sub[best_col].mean() * 100
            opt = sub[opt_col].mean()
            print(f'  {regime_labels[regime]:<14} {len(sub):>4} setups · '
                  f'{win:.1f}% win rate · {opt:+.1f}% avg opt return')

    # ── 7. Summary ────────────────────────────────────────────
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

    use_polygon = args.years > 0
    years       = args.years if args.years > 0 else 0
    tickers     = [args.ticker] if args.ticker else WATCHLIST

    if use_polygon and not os.environ.get('POLYGON_API_KEY'):
        print('⚠️  --years requires POLYGON_API_KEY to be set.')
        print('   Run: source ~/.zshrc')
        sys.exit(1)

    window_label = f'{years}-year Polygon window' if use_polygon else '60-day yfinance window'
    print('=' * 62)
    print(f'  ORB BACKTEST — {window_label}')
    print(f'  Gap filter: {args.min_gap:.1f}%–{args.max_gap:.1f}% · '
          f'Entry: candle {args.entry_candle} (9:{25+args.entry_candle*5:02d} ET)')
    print('=' * 62 + '\n')

    print('  Fetching VIX data...')
    vix_series = fetch_vix(years)
    print(f'  VIX: {len(vix_series)} daily bars loaded\n')

    all_trades = []
    for ticker in tickers:
        trades = backtest_ticker(ticker, args.min_gap, args.max_gap,
                                 args.entry_candle, use_polygon, years, vix_series)
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
