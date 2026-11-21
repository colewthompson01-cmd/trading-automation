#!/usr/bin/env python3
"""
swing_scanner.py — Weekly swing trade candidate screener
Runs Sunday evenings via GitHub Actions, emails top 8 S&P 500 setups.

Scoring:
  - Relative strength vs SPY (3-month)         30%
  - Momentum (price vs 50/20-day MA)           25%
  - Volume accumulation trend                  20%
  - RSI zone quality (45-65 is ideal)          15%
  - Pullback setup (not extended, off highs)   10%

Output: entry price, stop (-7%), target (+15%), time stop (10 trading days)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, os, smtplib, argparse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────
TOP_N          = 8
RS_WINDOW      = 63       # ~3 months for relative strength
VOLUME_WINDOW  = 30       # days for avg volume baseline
MIN_VOLUME     = 500_000  # minimum avg daily volume
MIN_PRICE      = 15
STOP_PCT       = 0.07     # 7% hard stop
TARGET_PCT     = 0.15     # 15% target
TIME_STOP_DAYS = 10       # exit if no movement in 10 trading days
MAX_SECTOR     = 2        # max picks per sector in top N

GMAIL_USER     = "cwthompsonusa@gmail.com"
GMAIL_TO       = "cwthompsonusa@gmail.com"
POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "swing_positions.json")
SNAPSHOT_FILE  = os.path.join(os.path.dirname(__file__), "latest_swing_scan.json")

# ── S&P 500 Universe ─────────────────────────────────────────
SP500 = [
    'MMM','AOS','ABT','ABBV','ACN','ADBE','AMD','AES','AFL','A','APD','ABNB','AKAM','ALB',
    'ARE','ALGN','ALLE','LNT','ALL','GOOGL','GOOG','MO','AMZN','AMCR','AEE','AAL','AEP',
    'AXP','AIG','AMT','AWK','AMP','AME','AMGN','APH','ADI','ANSS','AON','APA','AAPL','AMAT',
    'APTV','ACGL','ADM','ANET','AJG','AIZ','T','ATO','ADSK','ADP','AZO','AVB','AVY','AXON',
    'BKR','BALL','BAC','BK','BBWI','BAX','BDX','BRK-B','BBY','BIO','TECH','BIIB','BLK',
    'BX','BA','BKNG','BWA','BSX','BMY','AVGO','BR','BRO','BF-B','BLDR','BG','CDNS','CZR',
    'CPT','CPB','COF','CAH','KMX','CCL','CARR','CTLT','CAT','CBOE','CBRE','CDW','CE','COR',
    'CNC','CNP','CF','CHRW','CRL','SCHW','CHTR','CVX','CMG','CB','CHD','CI','CINF','CTAS',
    'CSCO','C','CFG','CLX','CME','CMS','KO','CTSH','CL','CMCSA','CAG','COP','ED','STZ',
    'CEG','COO','CPRT','GLW','CPAY','CTVA','CSGP','COST','CTRA','CRWD','CCI','CSX','CMI',
    'CVS','DHR','DRI','DVA','DAY','DECK','DE','DAL','DVN','DXCM','FANG','DLR','DFS','DG',
    'DLTR','D','DPZ','DOV','DOW','DHI','DTE','DUK','DD','EMN','ETN','EBAY','ECL','EIX',
    'EW','EA','ELV','EMR','ENPH','ETR','EOG','EPAM','EQT','EFX','EQIX','EQR','ESS','EL',
    'ETSY','EG','ES','EXC','EXPE','EXPD','EXR','XOM','FFIV','FDS','FICO','FAST','FRT',
    'FDX','FIS','FITB','FSLR','FE','FI','FLT','FMC','F','FTNT','FTV','FOXA','FOX','BEN',
    'FCX','GRMN','IT','GE','GEHC','GEV','GEN','GNRC','GD','GIS','GM','GPC','GILD','GS',
    'HAL','HIG','HAS','HCA','DOC','HSIC','HSY','HES','HPE','HLT','HD','HON','HRL','HST',
    'HWM','HPQ','HUBB','HUM','HBAN','HII','IBM','IEX','IDXX','ITW','INCY','IR','PODD',
    'INTC','ICE','IFF','IP','IPG','INTU','ISRG','IVZ','INVH','IQV','IRM','JBHT','JBL',
    'JKHY','J','JNJ','JCI','JPM','JNPR','K','KVUE','KDP','KEY','KEYS','KMB','KIM','KMI',
    'KKR','KLAC','KHC','KR','LHX','LH','LRCX','LW','LVS','LDOS','LEN','LLY','LIN','LYV',
    'LKQ','LMT','L','LOW','LULU','LYB','MTB','MRO','MPC','MKTX','MAR','MMC','MLM','MAS',
    'MA','MTCH','MKC','MCD','MCK','MDT','MRK','META','MET','MTD','MGM','MCHP','MU','MSFT',
    'MAA','MRNA','MHK','MOH','TAP','MDLZ','MPWR','MNST','MCO','MS','MOS','MSI','MSCI',
    'NDAQ','NTAP','NFLX','NEM','NWSA','NWS','NEE','NKE','NI','NDSN','NSC','NTRS','NOC',
    'NCLH','NRG','NUE','NVDA','NVR','NXPI','ORLY','OXY','ODFL','OMC','ON','OKE','ORCL',
    'OTIS','PCAR','PKG','PANW','PARA','PH','PAYX','PAYC','PYPL','PNR','PEP','PFE','PCG',
    'PM','PSX','PNW','PNC','POOL','PPG','PPL','PFG','PG','PGR','PLD','PRU','PEG','PTC',
    'PSA','PHM','QRVO','PWR','QCOM','DGX','RL','RJF','RTX','O','REG','REGN','RF','RSG',
    'RMD','RVTY','ROK','ROL','ROP','ROST','RCL','SPGI','CRM','SBAC','SLB','STX','SRE',
    'NOW','SHW','SPG','SWKS','SJM','SW','SNA','SOLV','SO','LUV','SWK','SBUX','STT','STLD',
    'STE','SYK','SMCI','SYF','SNPS','SYY','TMUS','TROW','TTWO','TPR','TRGP','TGT','TEL',
    'TDY','TFX','TER','TSLA','TXN','TXT','TMO','TJX','TSCO','TT','TDG','TRV','TRMB','TFC',
    'TYL','TSN','USB','UBER','UDR','ULTA','UNP','UAL','UPS','URI','UNH','UHS','VLO','VTR',
    'VLTO','VRSN','VRSK','VZ','VRTX','VTRS','VICI','V','VMC','WRB','GWW','WAB','WBA','WMT',
    'DIS','WBD','WM','WAT','WEC','WFC','WELL','WST','WDC','WY','WHR','WMB','WTW','WYNN',
    'XEL','XYL','YUM','ZBRA','ZBH','ZTS'
]

# ── Indicators ───────────────────────────────────────────────

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_bb_width(prices, period=20):
    sma   = prices.rolling(period).mean()
    sigma = prices.rolling(period).std()
    upper = sma + 2 * sigma
    lower = sma - 2 * sigma
    return (upper - lower) / sma * 100  # % width

# ── Score a single ticker ────────────────────────────────────

def score_ticker(ticker, spy_returns):
    """Returns scored candidate dict or None."""
    try:
        hist = yf.Ticker(ticker).history(period='1y')
        if hist.empty or len(hist) < RS_WINDOW + 10:
            return None

        c = hist['Close']
        v = hist['Volume']

        price     = float(c.iloc[-1])
        avg_vol   = float(v.tail(VOLUME_WINDOW).mean())

        if price < MIN_PRICE or avg_vol < MIN_VOLUME:
            return None

        # ── Relative Strength vs SPY ─────────────────────────
        # 3-month return percentile vs all S&P stocks (scored externally)
        if len(c) < RS_WINDOW:
            return None
        rs_return = (float(c.iloc[-1]) - float(c.iloc[-RS_WINDOW])) / float(c.iloc[-RS_WINDOW])
        spy_rs    = spy_returns.get('3m', 0)
        rs_excess = rs_return - spy_rs  # outperformance vs SPY

        # ── Momentum: price vs MAs ────────────────────────────
        sma20  = float(c.tail(20).mean()) if len(c) >= 20 else None
        sma50  = float(c.tail(50).mean()) if len(c) >= 50 else None
        sma200 = float(c.tail(200).mean()) if len(c) >= 200 else None

        if sma50 is None:
            return None

        above_50  = price > sma50
        above_200 = price > sma200 if sma200 else None
        pct_above_50 = (price - sma50) / sma50 * 100

        # Skip if too extended above 50-day (>15% = chasing)
        if pct_above_50 > 15:
            return None

        # ── Volume Accumulation ───────────────────────────────
        recent_vol  = float(v.tail(5).mean())
        vol_ratio   = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # ── RSI ───────────────────────────────────────────────
        rsi = calc_rsi(c)
        cur_rsi = float(rsi.iloc[-1])
        if np.isnan(cur_rsi):
            return None

        # ── Pullback Setup ────────────────────────────────────
        h20 = float(c.tail(20).max())
        pct_from_20d_high = (price - h20) / h20 * 100  # negative = pulled back

        h52 = float(c.tail(252).max())
        pct_from_52w_high = (price - h52) / h52 * 100

        # ── Composite Score ───────────────────────────────────
        score = 0.0

        # 1. Relative strength (30%) — excess return vs SPY, capped
        rs_score = min(30, max(0, (rs_excess + 0.05) * 150))
        score   += rs_score * W_RS / 30

        # 2. Momentum (25%) — above 50-day, 200-day bonus, not extended
        mom_score = 0.0
        if above_50:
            mom_score += 15
        if above_200:
            mom_score += 10
        # Penalize if very extended above 50-day
        if pct_above_50 > 10:
            mom_score -= (pct_above_50 - 10) * 2
        mom_score = max(0, min(25, mom_score))
        score    += mom_score * W_MOMENTUM / 25

        # 3. Volume (20%) — recent vol vs avg (1.2x+ = accumulation)
        vol_score = min(20, max(0, (vol_ratio - 0.8) * 20))
        score    += vol_score * W_VOLUME / 20

        # 4. RSI (15%) — 50-65 zone is ideal setup
        if 45 <= cur_rsi <= 65:
            rsi_score = 15 - abs(cur_rsi - 55) * 0.5
        elif cur_rsi < 45:
            rsi_score = max(0, (cur_rsi - 30) * 0.5)
        else:
            rsi_score = max(0, 15 - (cur_rsi - 65) * 1.0)
        score += rsi_score * W_RSI / 15

        # 5. Pullback setup (10%) — 3-8% off recent high is ideal entry
        if -8 <= pct_from_20d_high <= -2:
            setup_score = 10
        elif -15 <= pct_from_20d_high < -8:
            setup_score = 6
        elif -2 < pct_from_20d_high <= 0:
            setup_score = 4
        else:
            setup_score = 0
        score += setup_score * W_SETUP / 10

        # Normalize to 0-100
        score = round(min(100, max(0, score * 100)), 1)

        # Must be above 50-day to qualify
        if not above_50:
            return None

        # Must have positive RS vs SPY
        if rs_excess < -0.05:
            return None

        # RSI gates — not in extreme oversold or overbought
        if cur_rsi < 35 or cur_rsi > 75:
            return None

        # Entry levels
        stop   = round(price * (1 - STOP_PCT), 2)
        target = round(price * (1 + TARGET_PCT), 2)
        rr     = round(TARGET_PCT / STOP_PCT, 1)

        # Time stop: 10 trading days out
        time_stop = datetime.now() + timedelta(days=14)  # ~10 trading days

        return {
            'ticker':            ticker,
            'price':             round(price, 2),
            'score':             score,
            'rs_3m':             round(rs_return * 100, 1),
            'rs_excess':         round(rs_excess * 100, 1),
            'rsi':               round(cur_rsi, 1),
            'vol_ratio':         round(vol_ratio, 2),
            'pct_above_50':      round(pct_above_50, 1),
            'pct_from_20d_high': round(pct_from_20d_high, 1),
            'pct_from_52w_high': round(pct_from_52w_high, 1),
            'above_200':         above_200,
            'sma50':             round(sma50, 2),
            'sma200':            round(sma200, 2) if sma200 else None,
            'stop':              stop,
            'target':            target,
            'rr':                rr,
            'time_stop_date':    time_stop.strftime('%Y-%m-%d'),
        }
    except Exception:
        return None

# ── Fetch fundamentals for top candidates ────────────────────

def enrich_candidate(cand):
    """Add sector, name, analyst rating to a scored candidate."""
    try:
        info   = yf.Ticker(cand['ticker']).info
        cand['name']   = info.get('shortName', cand['ticker'])
        cand['sector'] = info.get('sector', 'Unknown')
        rec            = info.get('recommendationMean', None)
        cand['analyst_rating'] = round(rec, 1) if rec else None
        eps_gr = info.get('earningsGrowth', None)
        cand['eps_growth'] = f"{round(eps_gr*100,1)}%" if eps_gr else 'N/A'
        cand['fwd_pe']     = round(info.get('forwardPE', 0) or 0, 1) or 'N/A'
        return cand
    except Exception:
        cand['name']   = cand['ticker']
        cand['sector'] = 'Unknown'
        cand['analyst_rating'] = None
        cand['eps_growth'] = 'N/A'
        cand['fwd_pe']     = 'N/A'
        return cand

# ── Position Tracking ────────────────────────────────────────

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f:
            return json.load(f).get('positions', [])
    return []

def save_positions(positions):
    with open(POSITIONS_FILE, 'w') as f:
        json.dump({'positions': positions, 'updated': datetime.now().isoformat()}, f, indent=2)

def check_exits(positions):
    """Check open positions against stop/target/time stop. Returns (exits, still_open)."""
    exits      = []
    still_open = []
    today      = datetime.now()

    for pos in positions:
        if pos.get('status') != 'open':
            continue
        try:
            hist  = yf.Ticker(pos['ticker']).history(period='5d')
            price = float(hist['Close'].iloc[-1]) if not hist.empty else None
        except Exception:
            price = None

        if price is None:
            still_open.append(pos)
            continue

        entry      = pos['entry_price']
        stop       = pos['stop']
        target     = pos['target']
        time_stop  = datetime.strptime(pos['time_stop_date'], '%Y-%m-%d')
        pnl_pct    = round((price - entry) / entry * 100, 1)

        if price <= stop:
            exits.append({**pos, 'exit_price': price, 'exit_reason': 'STOP HIT', 'pnl_pct': pnl_pct})
        elif price >= target:
            exits.append({**pos, 'exit_price': price, 'exit_reason': 'TARGET HIT', 'pnl_pct': pnl_pct})
        elif today >= time_stop:
            exits.append({**pos, 'exit_price': price, 'exit_reason': 'TIME STOP', 'pnl_pct': pnl_pct})
        else:
            still_open.append({**pos, 'current_price': price, 'pnl_pct': pnl_pct})

    return exits, still_open

# ── Sector-Diversified Top N Selection ───────────────────────

def select_top_n(candidates, n=TOP_N):
    """Pick top N with max 2 per sector."""
    sector_count = {}
    selected     = []
    for c in sorted(candidates, key=lambda x: x['score'], reverse=True):
        sec = c.get('sector', 'Unknown')
        if sector_count.get(sec, 0) >= MAX_SECTOR:
            continue
        sector_count[sec] = sector_count.get(sec, 0) + 1
        selected.append(c)
        if len(selected) >= n:
            break
    return selected

# ── HTML Email ───────────────────────────────────────────────

def build_html(candidates, exits, still_open):
    now    = datetime.now().strftime('%A, %B %-d, %Y')
    n_cand = len(candidates)

    rows = ''
    for i, c in enumerate(candidates, 1):
        above_200_badge = '✓ Above 200d' if c.get('above_200') else '— Below 200d'
        above_200_color = '#1a7a1a' if c.get('above_200') else '#8B4513'
        analyst         = f"{c['analyst_rating']}/5" if c.get('analyst_rating') else '—'
        rr_color        = '#1a7a1a' if c['rr'] >= 2 else '#c88000'

        rows += f"""
        <tr style="background:{'#f9f9f9' if i%2==0 else '#ffffff'}">
          <td style="padding:12px 8px;font-weight:bold;font-size:15px">{i}. {c['ticker']}</td>
          <td style="padding:12px 8px;color:#555;font-size:12px">{c.get('name','')}<br>
              <span style="color:#888">{c.get('sector','')}</span></td>
          <td style="padding:12px 8px;text-align:center">
              <span style="font-size:16px;font-weight:bold">${c['price']}</span></td>
          <td style="padding:12px 8px;text-align:center">
              <span style="background:#1a4d1a;color:white;padding:3px 8px;border-radius:4px;font-size:13px;font-weight:bold">
              {c['score']}</span></td>
          <td style="padding:12px 8px;text-align:center;color:{'#1a7a1a' if c['rs_excess']>0 else '#c00000'};font-weight:bold">
              {'+' if c['rs_excess']>0 else ''}{c['rs_excess']}%</td>
          <td style="padding:12px 8px;text-align:center">{c['rsi']}</td>
          <td style="padding:12px 8px;text-align:center;color:#c00000;font-weight:bold">${c['stop']}</td>
          <td style="padding:12px 8px;text-align:center;color:#1a7a1a;font-weight:bold">${c['target']}</td>
          <td style="padding:12px 8px;text-align:center;color:{rr_color};font-weight:bold">{c['rr']}:1</td>
          <td style="padding:12px 8px;text-align:center;font-size:12px;color:{above_200_color}">{above_200_badge}</td>
          <td style="padding:12px 8px;text-align:center;font-size:12px">{c['pct_from_20d_high']}%<br>
              <span style="color:#888">off 20d high</span></td>
          <td style="padding:12px 8px;text-align:center;font-size:12px">{c['time_stop_date']}</td>
        </tr>"""

    exit_section = ''
    if exits:
        exit_rows = ''
        for e in exits:
            color  = '#1a7a1a' if e['pnl_pct'] > 0 else '#c00000'
            emoji  = '✅' if 'TARGET' in e['exit_reason'] else ('🛑' if 'STOP' in e['exit_reason'] else '⏱️')
            exit_rows += f"""
            <tr>
              <td style="padding:8px">{emoji} <b>{e['ticker']}</b></td>
              <td style="padding:8px">{e['exit_reason']}</td>
              <td style="padding:8px">${e['entry_price']} → ${e['exit_price']}</td>
              <td style="padding:8px;color:{color};font-weight:bold">{'+' if e['pnl_pct']>0 else ''}{e['pnl_pct']}%</td>
            </tr>"""

        exit_section = f"""
        <div style="margin:24px 0">
          <h2 style="color:#8B4513;font-size:14px;font-family:monospace;border-bottom:1px solid #ddd;padding-bottom:6px">
            EXIT ALERTS — ACTION REQUIRED</h2>
          <table style="width:100%;border-collapse:collapse;font-size:13px">
            <tr style="background:#fff3e0">
              <th style="padding:8px;text-align:left">Ticker</th>
              <th style="padding:8px;text-align:left">Reason</th>
              <th style="padding:8px;text-align:left">Price</th>
              <th style="padding:8px;text-align:left">P&L</th>
            </tr>
            {exit_rows}
          </table>
        </div>"""

    open_section = ''
    if still_open:
        open_rows = ''
        for p in still_open:
            color = '#1a7a1a' if p.get('pnl_pct', 0) > 0 else '#c00000'
            open_rows += f"""
            <tr>
              <td style="padding:6px"><b>{p['ticker']}</b></td>
              <td style="padding:6px">${p['entry_price']}</td>
              <td style="padding:6px">${p.get('current_price','—')}</td>
              <td style="padding:6px;color:{color}">{'+' if p.get('pnl_pct',0)>0 else ''}{p.get('pnl_pct','—')}%</td>
              <td style="padding:6px;color:#c00000">${p['stop']}</td>
              <td style="padding:6px;color:#1a7a1a">${p['target']}</td>
              <td style="padding:6px;color:#888;font-size:11px">{p['time_stop_date']}</td>
            </tr>"""

        open_section = f"""
        <div style="margin:24px 0">
          <h2 style="color:#333;font-size:14px;font-family:monospace;border-bottom:1px solid #ddd;padding-bottom:6px">
            OPEN POSITIONS ({len(still_open)})</h2>
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <tr style="background:#f0f0f0">
              <th style="padding:6px;text-align:left">Ticker</th>
              <th style="padding:6px">Entry</th>
              <th style="padding:6px">Current</th>
              <th style="padding:6px">P&L</th>
              <th style="padding:6px">Stop</th>
              <th style="padding:6px">Target</th>
              <th style="padding:6px">Time Stop</th>
            </tr>
            {open_rows}
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:20px;font-family:Arial,sans-serif;background:#f5f5f5">
<div style="max-width:900px;margin:0 auto;background:white;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1)">

  <div style="background:#1a1a2e;color:white;padding:20px 24px">
    <div style="font-family:monospace;font-size:11px;color:#aaa">WEEKLY SWING SCANNER</div>
    <div style="font-size:22px;font-weight:bold;margin:4px 0">📈 Top {n_cand} Swing Setups</div>
    <div style="font-size:13px;color:#ccc">{now} · S&P 500 · 1–2 week horizon</div>
    <div style="margin-top:12px;font-size:12px;color:#aaa;line-height:1.6">
      Entry at current price · Stop −7% · Target +15% · Exit if no move in 10 trading days
    </div>
  </div>

  <div style="padding:20px 24px">
    {exit_section}
    {open_section}

    <h2 style="color:#333;font-size:14px;font-family:monospace;border-bottom:1px solid #ddd;padding-bottom:6px">
      NEW CANDIDATES THIS WEEK</h2>

    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:#1a1a2e;color:white">
        <th style="padding:10px 8px;text-align:left">Ticker</th>
        <th style="padding:10px 8px;text-align:left">Name / Sector</th>
        <th style="padding:10px 8px;text-align:center">Price</th>
        <th style="padding:10px 8px;text-align:center">Score</th>
        <th style="padding:10px 8px;text-align:center">RS vs SPY</th>
        <th style="padding:10px 8px;text-align:center">RSI</th>
        <th style="padding:10px 8px;text-align:center">Stop</th>
        <th style="padding:10px 8px;text-align:center">Target</th>
        <th style="padding:10px 8px;text-align:center">R:R</th>
        <th style="padding:10px 8px;text-align:center">Trend</th>
        <th style="padding:10px 8px;text-align:center">Setup</th>
        <th style="padding:10px 8px;text-align:center">Time Stop</th>
      </tr>
      {rows}
    </table>
    </div>

    <div style="margin-top:20px;padding:12px;background:#f9f9f9;border-radius:6px;font-size:11px;color:#888;line-height:1.6">
      ⚠️ Not investment advice. Always confirm entry with your own analysis before trading.
      Stop losses are suggested levels — execute them without hesitation.
      Score = composite of relative strength (30%), momentum (25%), volume (20%), RSI (15%), setup (10%).
    </div>
  </div>
</div>
</body></html>"""

# ── Send Email ───────────────────────────────────────────────

def send_email(html, n_cand, n_exits):
    password = os.environ.get('GMAIL_APP_PASSWORD')
    if not password:
        print('[email] GMAIL_APP_PASSWORD not set — skipping email')
        return

    subject = f"📈 Swing Scanner: {n_cand} setups"
    if n_exits:
        subject += f" · {n_exits} EXIT ALERT{'S' if n_exits>1 else ''}"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = GMAIL_USER
    msg['To']      = GMAIL_TO
    msg.attach(MIMEText(html, 'html'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
        s.login(GMAIL_USER, password)
        s.send_message(msg)
    print(f'[email] Sent: {subject}')

# ── Write JSON Snapshot ──────────────────────────────────────

def write_snapshot(candidates, exits, still_open):
    snap = {
        'scan_time':   datetime.now().isoformat(),
        'candidates':  candidates,
        'exits':       exits,
        'open_positions': still_open,
    }
    with open(SNAPSHOT_FILE, 'w') as f:
        json.dump(snap, f, indent=2)
    print(f'[snapshot] Written → {SNAPSHOT_FILE}')

# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', action='store_true', help='Send email')
    parser.add_argument('--no-snapshot', action='store_true')
    args = parser.parse_args()

    print('=' * 58)
    print('  SWING SCANNER — S&P 500')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 58)

    # Fetch SPY 3-month return as RS baseline
    print('\n[1] Fetching SPY baseline...')
    try:
        spy_hist  = yf.Ticker('SPY').history(period='6mo')
        spy_3m    = (float(spy_hist['Close'].iloc[-1]) - float(spy_hist['Close'].iloc[-RS_WINDOW])) / float(spy_hist['Close'].iloc[-RS_WINDOW])
        spy_returns = {'3m': spy_3m}
        print(f'    SPY 3-month return: {round(spy_3m*100,1)}%')
    except Exception as e:
        print(f'    SPY fetch failed: {e}')
        spy_returns = {'3m': 0.0}

    # Score all tickers
    print(f'\n[2] Scoring {len(SP500)} tickers...')
    candidates = []
    for i, ticker in enumerate(SP500):
        if i % 100 == 0:
            print(f'    [{i:>3}/{len(SP500)}] candidates: {len(candidates)}')
        result = score_ticker(ticker, spy_returns)
        if result:
            candidates.append(result)

    print(f'\n    Raw candidates: {len(candidates)}')

    if not candidates:
        print('    No candidates found this week.')
        return

    # Pre-select top 25 by raw score for enrichment (keeps API calls manageable)
    pre_select = sorted(candidates, key=lambda x: x['score'], reverse=True)[:25]

    # Enrich with fundamentals (sector + name + analyst data)
    print(f'\n[3] Enriching top {len(pre_select)} candidates with fundamentals...')
    enriched = [enrich_candidate(c) for c in pre_select]

    # Now apply sector diversification on enriched candidates
    top = select_top_n(enriched, TOP_N)

    print('\n  TOP SWING CANDIDATES:')
    for c in top:
        print(f"    {c['ticker']:6} | Score {c['score']:5} | RS+{c['rs_excess']}% | "
              f"RSI {c['rsi']} | Stop ${c['stop']} | Target ${c['target']} | {c.get('sector','')}")

    # Check open positions for exits
    print('\n[4] Checking open positions...')
    positions  = load_positions()
    exits, still_open = check_exits(positions) if positions else ([], [])

    if exits:
        print(f'    EXIT ALERTS: {len(exits)}')
        for e in exits:
            print(f"    {e['ticker']} → {e['exit_reason']} at ${e.get('exit_price','?')} ({e.get('pnl_pct','?')}%)")
        # Update positions file — keep still_open only
        save_positions(still_open)
    else:
        print('    No exits triggered.')

    # Write snapshot
    if not args.no_snapshot:
        write_snapshot(top, exits, still_open)

    # Send email
    if args.email:
        html = build_html(top, exits, still_open)
        send_email(html, len(top), len(exits))

    print(f'\n{"=" * 58}')
    print(f'  Done. {len(top)} candidates, {len(exits)} exits, {len(still_open)} open.')
    print(f'{"=" * 58}\n')

# Scoring weight constants (defined here for reference in score_ticker)
W_RS       = 0.30
W_MOMENTUM = 0.25
W_VOLUME   = 0.20
W_RSI      = 0.15
W_SETUP    = 0.10

if __name__ == '__main__':
    main()
