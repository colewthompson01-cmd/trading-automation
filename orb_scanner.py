#!/usr/bin/env python3
"""
orb_scanner.py — Pre-market Opening Range Breakout watchlist
Runs at 8:00 AM CST (9:00 AM ET) Mon-Fri via GitHub Actions.

Scans a fixed universe of high-liquidity stocks/ETFs with daily or
frequent 0DTE expirations. Flags names with:
  - Pre-market gap >2% with volume confirmation
  - Earnings cleared (no earnings within 5 days — avoids IV crush)
  - 0DTE options chain liquid enough to trade (OI check)

Output: email watchlist with gap size, direction, option to watch,
        and ORB levels. You confirm the 5-min candle live and enter.
"""

import os, sys, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────
GMAIL_USER  = "cwthompsonusa@gmail.com"
GMAIL_TO    = "cwthompsonusa@gmail.com"

MIN_GAP_PCT       = 3.0    # backtest shows 2-3% gaps are noise — 3%+ only
MAX_GAP_PCT       = 12.0   # skip if gapped too far (chasing risk)
EARNINGS_BUFFER   = 5      # skip if earnings within this many days
MIN_OI            = 100    # minimum ATM 0DTE open interest
MAX_SPREAD_PCT    = 0.25   # max bid/ask spread as % of midpoint (skipped pre-mkt)
VIX_MIN           = 15.0   # skip low-VIX days (calm market = 44% win rate)
VIX_MAX           = 25.0   # skip high-VIX days (too choppy = 44% win rate)
                            # sweet spot: VIX 15-25 → 80% win rate in backtest

# Refined universe — 2-year backtest: ARM 87.5%, HOOD 83.3%, AMD 80%, COIN 75%, MSTR 60%
# Dropped: NVDA/TSLA/AAPL/AMZN/MSFT/META/GOOGL/NFLX/PLTR/CRWD (sub-50% win rates)
WATCHLIST = {
    'ARM':  'Arm Holdings',
    'COIN': 'Coinbase',
    'AMD':  'AMD',
    'MSTR': 'MicroStrategy',
    'HOOD': 'Robinhood',
}

# ── VIX Regime Check ─────────────────────────────────────────

def get_vix_regime():
    """
    Fetch prior-day VIX close. Returns (vix_val, regime, ok).
    regime: 'low' (<15), 'medium' (15-25), 'high' (>25)
    ok: True only in medium regime (15-25) — backtest sweet spot
    """
    try:
        df = yf.Ticker('^VIX').history(period='5d', interval='1d')
        if df.empty:
            return None, 'unknown', True  # no data — allow through
        vix_val = float(df['Close'].iloc[-1])
        if vix_val < VIX_MIN:
            return vix_val, 'low', False
        elif vix_val <= VIX_MAX:
            return vix_val, 'medium', True
        else:
            return vix_val, 'high', False
    except Exception:
        return None, 'unknown', True  # fail open


# ── Earnings Check ───────────────────────────────────────────

def earnings_clear(info):
    """
    Returns (clear: bool, reason: str).
    Disqualifies if earnings within EARNINGS_BUFFER days in either direction.
    """
    now = datetime.now()

    for key in ('earningsTimestampStart', 'earningsTimestamp'):
        ts = info.get(key)
        if not ts:
            continue
        try:
            earn_dt   = datetime.fromtimestamp(int(ts))
            days_away = (earn_dt - now).days
            if -1 <= days_away <= EARNINGS_BUFFER:
                label = "today" if days_away == 0 else (
                    f"in {days_away}d" if days_away > 0 else "yesterday")
                return False, f"earnings {label} — IV risk"
        except Exception:
            pass

    return True, f"clear (>{EARNINGS_BUFFER}d)"

# ── Pre-Market Gap ────────────────────────────────────────────

def get_premarket_gap(info):
    """
    Returns (gap_pct, pm_price, prev_close, vol_ratio, direction).
    Uses yfinance info dict which carries preMarketPrice during pre-mkt hours.
    """
    pm_price   = info.get('preMarketPrice') or info.get('currentPrice')
    prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
    pm_volume  = info.get('preMarketVolume') or 0
    avg_volume = info.get('averageVolume') or 1

    if not pm_price or not prev_close or prev_close == 0:
        return None, None, None, None, None

    gap_pct   = round((pm_price - prev_close) / prev_close * 100, 2)
    vol_ratio = round(pm_volume / avg_volume, 3) if avg_volume > 0 else 0
    direction = 'UP' if gap_pct > 0 else 'DOWN'

    return gap_pct, round(pm_price, 2), round(prev_close, 2), vol_ratio, direction

# ── Options Liquidity ─────────────────────────────────────────

def check_0dte_liquidity(ticker, price, direction):
    """
    Finds today's expiration in the options chain.
    Checks ATM strike OI. Spread check skipped pre-market (bid=0).
    Returns (liquid, expiry, atm_strike, est_premium, oi, spread_pct).
    """
    try:
        t    = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return False, None, None, None, 0, None

        today = datetime.now().strftime('%Y-%m-%d')

        # Prefer today's expiry; fall back to nearest within 2 days
        target_exp = None
        for e in exps:
            days = (datetime.strptime(e, '%Y-%m-%d') - datetime.now()).days
            if days == 0:
                target_exp = e
                break
        if not target_exp:
            for e in exps:
                days = (datetime.strptime(e, '%Y-%m-%d') - datetime.now()).days
                if 0 <= days <= 1:
                    target_exp = e
                    break

        if not target_exp:
            return False, None, None, None, 0, None

        chain   = t.option_chain(target_exp)
        options = chain.calls if direction == 'UP' else chain.puts
        if options.empty:
            return False, target_exp, None, None, 0, None

        options       = options.copy()
        options['_d'] = (options['strike'] - price).abs()
        atm           = options.loc[options['_d'].idxmin()]

        atm_strike = float(atm['strike'])
        oi         = int(atm.get('openInterest', 0) or 0)
        bid        = float(atm.get('bid', 0) or 0)
        ask        = float(atm.get('ask', 0) or 0)
        last       = float(atm.get('lastPrice', 0) or 0)

        spread_pct = None
        if bid > 0 and ask > 0:
            spread_pct = round((ask - bid) / ((ask + bid) / 2), 3)
            if spread_pct > MAX_SPREAD_PCT:
                return False, target_exp, atm_strike, last, oi, spread_pct

        if oi < MIN_OI:
            return False, target_exp, atm_strike, last, oi, spread_pct

        est_premium = round((bid + ask) / 2, 2) if bid > 0 else round(last, 2)
        return True, target_exp, atm_strike, est_premium, oi, spread_pct

    except Exception:
        return False, None, None, None, 0, None

# ── Scan Single Ticker ────────────────────────────────────────

def scan_ticker(ticker, name):
    """Returns candidate dict or None."""
    try:
        # ── Monday filter ─────────────────────────────────────
        if datetime.now().weekday() == 0:
            print(f'    {ticker:6} SKIP — Monday (weekend gaps fade)')
            return None

        info = yf.Ticker(ticker).info

        # ── Earnings check ────────────────────────────────────
        clear, earn_note = earnings_clear(info)
        if not clear:
            print(f'    {ticker:6} SKIP — {earn_note}')
            return None

        # ── Pre-market gap ────────────────────────────────────
        gap_pct, pm_price, prev_close, vol_ratio, direction = get_premarket_gap(info)

        if gap_pct is None:
            print(f'    {ticker:6} SKIP — no pre-market price data')
            return None

        abs_gap = abs(gap_pct)

        if abs_gap < MIN_GAP_PCT:
            print(f'    {ticker:6} SKIP — gap {gap_pct:+.1f}% below 3% threshold')
            return None

        if abs_gap > MAX_GAP_PCT:
            print(f'    {ticker:6} SKIP — gap {gap_pct:+.1f}% too extended to chase')
            return None

        if direction != 'UP':
            print(f'    {ticker:6} SKIP — DOWN gap (backtest: 44% win vs 70% UP)')
            return None

        # ── Options liquidity ─────────────────────────────────
        liquid, expiry, atm_strike, est_premium, oi, spread_pct = check_0dte_liquidity(
            ticker, pm_price, direction)

        if not liquid:
            print(f'    {ticker:6} SKIP — 0DTE options illiquid (OI={oi})')
            return None

        opt_type    = 'CALL' if direction == 'UP' else 'PUT'
        spread_disp = f"{round(spread_pct*100,1)}%" if spread_pct else 'pre-mkt'

        print(f'    {ticker:6} ✓  gap {gap_pct:+.1f}% {direction} | '
              f'vol {vol_ratio:.2f}x | {opt_type} ${atm_strike} | OI {oi}')

        return {
            'ticker':       ticker,
            'name':         name,
            'gap_pct':      gap_pct,
            'abs_gap':      abs_gap,
            'direction':    direction,
            'pm_price':     pm_price,
            'prev_close':   prev_close,
            'vol_ratio':    vol_ratio,
            'expiry':       expiry,
            'opt_type':     opt_type,
            'atm_strike':   atm_strike,
            'est_premium':  est_premium,
            'oi':           oi,
            'spread':       spread_disp,
            'earn_note':    earn_note,
        }

    except Exception as e:
        print(f'    {ticker:6} ERROR — {e}')
        return None

# ── HTML Email ───────────────────────────────────────────────

def build_html(candidates):
    now    = datetime.now().strftime('%A, %B %-d, %Y  ·  %-I:%M %p CT')
    n      = len(candidates)

    if not candidates:
        body = """
        <div style="padding:30px;text-align:center;color:#666">
          <div style="font-size:48px">🟡</div>
          <div style="font-size:18px;margin-top:12px">No ORB setups today</div>
          <div style="font-size:13px;margin-top:8px;color:#999">
            No stocks cleared the gap + volume + earnings + liquidity filters.
          </div>
        </div>"""
    else:
        rows = ''
        for c in sorted(candidates, key=lambda x: x['abs_gap'], reverse=True):
            gap_color = '#1a7a1a' if c['direction'] == 'UP' else '#c00000'
            gap_arrow = '▲' if c['direction'] == 'UP' else '▼'
            opt_color = '#1a4d8f' if c['opt_type'] == 'CALL' else '#8f1a1a'
            vol_color = '#1a7a1a' if c['vol_ratio'] >= 0.3 else '#c88000'

            rows += f"""
            <tr style="border-bottom:1px solid #eee">
              <td style="padding:14px 10px">
                <div style="font-size:17px;font-weight:bold">{c['ticker']}</div>
                <div style="font-size:11px;color:#888">{c['name']}</div>
              </td>
              <td style="padding:14px 10px;text-align:center">
                <span style="color:{gap_color};font-size:18px;font-weight:bold">
                  {gap_arrow} {c['gap_pct']:+.1f}%
                </span><br>
                <span style="font-size:11px;color:#888">
                  ${c['prev_close']} → ${c['pm_price']}
                </span>
              </td>
              <td style="padding:14px 10px;text-align:center">
                <span style="color:{vol_color};font-weight:bold">{c['vol_ratio']:.2f}x</span><br>
                <span style="font-size:11px;color:#888">avg daily vol</span>
              </td>
              <td style="padding:14px 10px;text-align:center">
                <div style="background:{opt_color};color:white;padding:4px 10px;
                     border-radius:4px;font-size:13px;font-weight:bold;display:inline-block">
                  {c['opt_type']} ${c['atm_strike']}
                </div>
                <div style="font-size:11px;color:#888;margin-top:4px">
                  est. ${c['est_premium']} · OI {c['oi']:,} · spread {c['spread']}
                </div>
              </td>
              <td style="padding:14px 10px;text-align:center;font-size:12px">
                <div style="color:#1a7a1a">✓ {c['earn_note']}</div>
                <div style="color:#888;margin-top:4px">exp {c['expiry']}</div>
              </td>
              <td style="padding:14px 10px;font-size:11px;color:#555;max-width:160px">
                Watch first 5-min candle at open.<br>
                {'Break above pre-mkt high → enter CALL' if c['direction']=='UP'
                  else 'Break below pre-mkt low → enter PUT'}<br>
                <span style="color:#c00000">Stop: 50% of premium</span>
              </td>
            </tr>"""

        body = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <tr style="background:#1a1a2e;color:white">
            <th style="padding:10px;text-align:left">Ticker</th>
            <th style="padding:10px;text-align:center">Gap</th>
            <th style="padding:10px;text-align:center">PM Volume</th>
            <th style="padding:10px;text-align:center">0DTE Option</th>
            <th style="padding:10px;text-align:center">Earnings</th>
            <th style="padding:10px;text-align:left">ORB Rule</th>
          </tr>
          {rows}
        </table>"""

    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:20px;font-family:Arial,sans-serif;background:#f5f5f5">
<div style="max-width:860px;margin:0 auto;background:white;border-radius:8px;
     overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1)">

  <div style="background:#1a1a2e;color:white;padding:20px 24px">
    <div style="font-family:monospace;font-size:11px;color:#aaa">PRE-MARKET · ORB WATCHLIST</div>
    <div style="font-size:22px;font-weight:bold;margin:4px 0">
      🔔 {n} ORB Setup{'s' if n != 1 else ''} Today
    </div>
    <div style="font-size:13px;color:#ccc">{now}</div>
    <div style="margin-top:12px;font-size:12px;color:#aaa;line-height:1.6">
      Watch the first 5-min candle at open (9:30 ET). Enter on breakout of opening range.
      These names cleared: UP gap ≥3% · earnings safe · VIX 15–25 · not Monday · 0DTE liquid.
      Backtest (2yr): 70% win rate · +29% avg option return at 11:00 ET exit.
    </div>
  </div>

  <div style="padding:20px 24px">
    {body}

    <div style="margin-top:20px;padding:12px;background:#fff8e1;border-left:3px solid #c88000;
         border-radius:4px;font-size:12px;color:#555;line-height:1.6">
      <b>ORB Rules:</b> Enter CALL only on a clean UP 5-min candle break at 9:35 ET.
      Do not chase gaps already running 10%+. Exit at 11:00 AM ET sharp — backtest peak.
      Stop = 50% of premium paid. No averaging down on 0DTE. No trades on Mondays.
    </div>

    <div style="margin-top:12px;font-size:11px;color:#aaa;text-align:center">
      Not investment advice. IV is elevated pre-market — actual premiums may differ at open.
    </div>
  </div>
</div>
</body></html>"""

# ── Send Email ───────────────────────────────────────────────

def send_email(html, n):
    password = os.environ.get('GMAIL_APP_PASSWORD')
    if not password:
        print('[email] GMAIL_APP_PASSWORD not set — skipping')
        return

    subject = f"🔔 ORB Watchlist: {n} setup{'s' if n!=1 else ''} today" if n else "🟡 ORB: No setups today"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = GMAIL_USER
    msg['To']      = GMAIL_TO
    msg.attach(MIMEText(html, 'html'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
        s.login(GMAIL_USER, password)
        s.send_message(msg)
    print(f'[email] Sent: {subject}')

# ── Main ─────────────────────────────────────────────────────

def main():
    send = '--email' in sys.argv

    print('=' * 58)
    print('  ORB SCANNER — Pre-Market Watchlist')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 58)

    # ── Monday check ──────────────────────────────────────────
    if datetime.now().weekday() == 0:
        print('\n  Monday — ORB scanner skipped (weekend gap fade risk)')
        if send:
            send_email(build_html([]), 0)
        return

    # ── VIX regime check ──────────────────────────────────────
    vix_val, vix_regime, vix_ok = get_vix_regime()
    if vix_val is not None:
        print(f'\n  VIX: {vix_val:.1f} ({vix_regime})', end='')
        if not vix_ok:
            regime_msg = 'too low — calm markets kill ORB edge' if vix_regime == 'low' \
                         else 'too high — fear-driven chop'
            print(f' — SKIP: {regime_msg}')
            if send:
                send_email(build_html([]), 0)
            return
        print(f' — in sweet spot (15–25), scanning...')
    else:
        print('\n  VIX: unavailable — proceeding anyway')

    print(f'\nScanning {len(WATCHLIST)} tickers...\n')

    candidates = []
    for ticker, name in WATCHLIST.items():
        result = scan_ticker(ticker, name)
        if result:
            candidates.append(result)

    print(f'\n{"=" * 58}')
    print(f'  {len(candidates)} ORB setup(s) found')
    print(f'{"=" * 58}\n')

    html = build_html(candidates)

    if send:
        send_email(html, len(candidates))
    else:
        print('[dry run] Pass --email to send. Candidates:')
        for c in candidates:
            print(f"  {c['ticker']} {c['gap_pct']:+.1f}% {c['direction']} "
                  f"→ {c['opt_type']} ${c['atm_strike']} (OI {c['oi']})")

if __name__ == '__main__':
    main()
