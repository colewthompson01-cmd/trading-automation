#!/usr/bin/env python3
"""
swing_scanner.py — Weekly pre-breakout candidate screener
Runs Sunday evenings via GitHub Actions, emails top pick for the week.

Scoring (data-driven from 7-year backtest, 1,586 trades):
  - 50MA vs 200MA spread                       25%  strongest separator
  - 6-month RS vs SPY                          20%  long-duration leadership
  - Prior 4-week quietness (not running yet)   20%  reward flat/slight pullback
  - ATR compression (tight base forming)       15%  VCP indicator
  - RSI zone 40-55                             10%  not overbought
  - 1-year RS vs SPY                           10%  very long duration RS

Hard filters: above 200MA · 50MA > 200MA · RSI < 68 · prior 4wk return < 15%
Quality gates: composite score ≥ 60 · 6M RS vs SPY ≥ +40%

Output: 1 pick/week · stop -10% · target +30% · avg exit ~68 days · EV +55% at 1yr
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
TOP_N          = 1         # single best setup per week
MIN_SCORE      = 60        # minimum composite score (0-100)
MIN_RS6M       = 0.40      # minimum 6M RS vs SPY (+40%) — backtest optimal
MIN_VOLUME     = 500_000
MIN_PRICE      = 15
STOP_PCT       = 0.10      # 10% stop
TARGET_PCT     = 0.30      # 30% target (~68 days avg to hit)
TIME_STOP_DAYS = 252       # 1-year max hold (rarely needed)
MAX_SECTOR     = 2

# Scoring weights (sum to 1.0)
W_TREND_SPREAD = 0.25   # 50MA vs 200MA gap
W_RS_6M        = 0.20   # 6-month RS vs SPY
W_QUIETNESS    = 0.20   # prior 4-week NOT running
W_ATR_COMPRESS = 0.15   # ATR compression
W_RSI          = 0.10   # RSI zone 40-55
W_RS_1Y        = 0.10   # 1-year RS vs SPY

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
    """
    Data-driven pre-breakout score. Signals ranked by research finding strength:
      1. 50MA vs 200MA spread  (25%) — strongest separator in 7-year study
      2. 6M RS vs SPY          (20%) — long-duration leadership
      3. Prior 4-week quietness(20%) — not already running
      4. ATR compression       (15%) — tight base forming
      5. RSI zone 40-55        (10%) — not overbought
      6. 1Y RS vs SPY          (10%) — very long duration RS
    """
    try:
        hist = yf.Ticker(ticker).history(period='2y')
        if hist.empty or len(hist) < 210:
            return None

        c   = hist['Close']
        v   = hist['Volume']
        price     = float(c.iloc[-1])
        avg_vol   = float(v.tail(50).mean())

        if price < MIN_PRICE or avg_vol < MIN_VOLUME:
            return None

        # ── MAs ───────────────────────────────────────────────
        if len(c) < 200:
            return None
        ma50  = float(c.tail(50).mean())
        ma200 = float(c.tail(200).mean())

        # ── Hard filters ──────────────────────────────────────
        # Must be above 200MA and 50MA must be above 200MA
        if price < ma200:
            return None
        if ma50 < ma200:
            return None

        # RSI — must not be overbought
        rsi_s  = calc_rsi(c)
        cur_rsi = float(rsi_s.iloc[-1])
        if np.isnan(cur_rsi) or cur_rsi > 68:
            return None

        # Prior 4-week return — if already running >15%, skip
        ret_4w = float(c.iloc[-1] / c.iloc[-20] - 1) if len(c) >= 20 else 0
        if ret_4w > 0.15:
            return None

        # ── Signal 1: 50MA vs 200MA spread (25%) ─────────────
        # Bigger gap = healthier uptrend. Normalize: 5% gap = 50 pts, 20% = 100
        ma_spread     = (ma50 - ma200) / ma200   # e.g. 0.12 = 12% above
        trend_score   = min(100, max(0, ma_spread / 0.20 * 100))

        # ── Signal 2: 6-month RS vs SPY (20%) ────────────────
        rs_6m_val  = spy_returns.get('6m_stock_ret', {}).get(ticker, None)
        spy_6m     = spy_returns.get('6m', 0)
        rs_6m      = None
        rs_6m_score = 50
        if len(c) >= 126:
            rs_6m       = float(c.iloc[-1] / c.iloc[-126] - 1) - spy_6m
            # Sweet spot: +10% to +40% above SPY. Research: 0 to +30% = best breakout rate
            rs_6m_score = min(100, max(0, (rs_6m + 0.05) / 0.35 * 100))

        # ── Signal 3: Prior 4-week quietness (20%) ────────────
        # Research finding: breakouts had +0.03% prior 4wk vs +3.44% for non-events
        # Score peaks at flat to slightly down (-5% to +2%), degrades as it rises
        if ret_4w <= 0.02:
            quiet_score = 100   # flat or slightly down = perfect setup
        elif ret_4w <= 0.08:
            quiet_score = max(0, 100 - (ret_4w - 0.02) / 0.06 * 70)
        else:
            quiet_score = max(0, 30 - (ret_4w - 0.08) / 0.07 * 30)
        # Also reward slight pullback (down 3-10% = healthy reset)
        if -0.10 <= ret_4w <= -0.03:
            quiet_score = min(100, quiet_score + 15)

        # ── Signal 4: ATR compression (15%) ──────────────────
        # Recent ATR (10d) vs prior ATR (10-30d ago). Compression < 1 = tight base
        tr = pd.concat([
            hist['High'] - hist['Low'],
            (hist['High'] - hist['Close'].shift()).abs(),
            (hist['Low']  - hist['Close'].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_recent = float(tr.tail(10).mean())
        atr_prior  = float(tr.iloc[-30:-10].mean()) if len(tr) >= 30 else float(tr.mean())
        compression = atr_recent / atr_prior if atr_prior > 0 else 1.0
        # < 0.6 = very tight (best), > 1.2 = expanding (worst)
        atr_score = min(100, max(0, (1.2 - compression) / 0.6 * 100))

        # ── Signal 5: RSI zone 40-55 (10%) ───────────────────
        # Research: RSI < 50 bucket had highest breakout rate (27.1%)
        if 40 <= cur_rsi <= 55:
            rsi_score = 100
        elif 35 <= cur_rsi < 40 or 55 < cur_rsi <= 62:
            rsi_score = 60
        elif 62 < cur_rsi <= 68:
            rsi_score = 20
        else:
            rsi_score = 0

        # ── Signal 6: 1-year RS vs SPY (10%) ─────────────────
        spy_1y = spy_returns.get('1y', 0)
        rs_1y  = None
        rs_1y_score = 50
        if len(c) >= 252:
            rs_1y       = float(c.iloc[-1] / c.iloc[-252] - 1) - spy_1y
            rs_1y_score = min(100, max(0, (rs_1y + 0.05) / 0.50 * 100))

        # ── Composite (0-100) ─────────────────────────────────
        composite = (
            trend_score  * W_TREND_SPREAD +
            rs_6m_score  * W_RS_6M +
            quiet_score  * W_QUIETNESS +
            atr_score    * W_ATR_COMPRESS +
            rsi_score    * W_RSI +
            rs_1y_score  * W_RS_1Y
        )
        score = round(composite, 1)

        h52  = float(c.tail(252).max())
        h20  = float(c.tail(20).max())
        stop   = round(price * (1 - STOP_PCT), 2)
        target = round(price * (1 + TARGET_PCT), 2)
        rr     = round(TARGET_PCT / STOP_PCT, 1)
        time_stop = datetime.now() + timedelta(days=365)  # 1-year max hold

        return {
            'ticker':            ticker,
            'price':             round(price, 2),
            'score':             score,
            'rs_6m':             round(rs_6m * 100, 1) if rs_6m is not None else None,
            'rs_1y':             round(rs_1y * 100, 1) if rs_1y is not None else None,
            'rsi':               round(cur_rsi, 1),
            'ret_4w':            round(ret_4w * 100, 1),
            'ma_spread':         round(ma_spread * 100, 1),
            'atr_compression':   round(compression, 2),
            'pct_from_52w_high': round((price - h52) / h52 * 100, 1),
            'pct_from_20d_high': round((price - h20) / h20 * 100, 1),
            'above_200':         True,
            'ma50':              round(ma50, 2),
            'ma200':             round(ma200, 2),
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
    """Pick top N with max 2 per sector, applying score and RS6M filters."""
    sector_count = {}
    selected     = []
    for c in sorted(candidates, key=lambda x: x['score'], reverse=True):
        if c['score'] < MIN_SCORE:
            continue
        rs6m_raw = (c.get('rs_6m') or 0) / 100  # stored as %, compare as decimal
        if rs6m_raw < MIN_RS6M:
            continue
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
        rr_color   = '#1a7a1a' if c['rr'] >= 2 else '#c88000'
        rs6m       = c.get('rs_6m', 0) or 0
        rs6m_color = '#1a7a1a' if rs6m > 0 else '#c00000'
        ma_spread  = c.get('ma_spread', 0) or 0
        atr_cmp    = c.get('atr_compression', 1.0) or 1.0
        atr_color  = '#1a7a1a' if atr_cmp < 0.85 else ('#c88000' if atr_cmp < 1.0 else '#888')
        ret4w      = c.get('ret_4w', 0) or 0
        ret4w_color= '#1a7a1a' if ret4w <= 2 else ('#c88000' if ret4w <= 8 else '#c00000')

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
          <td style="padding:12px 8px;text-align:center;color:{rs6m_color};font-weight:bold">
              {'+' if rs6m>0 else ''}{rs6m}%</td>
          <td style="padding:12px 8px;text-align:center;font-weight:bold">
              {'+' if ma_spread>0 else ''}{ma_spread}%</td>
          <td style="padding:12px 8px;text-align:center;color:{atr_color};font-weight:bold">
              {atr_cmp:.2f}</td>
          <td style="padding:12px 8px;text-align:center">{c['rsi']}</td>
          <td style="padding:12px 8px;text-align:center;color:{ret4w_color}">
              {'+' if ret4w>0 else ''}{ret4w}%</td>
          <td style="padding:12px 8px;text-align:center;color:#c00000;font-weight:bold">${c['stop']}</td>
          <td style="padding:12px 8px;text-align:center;color:#1a7a1a;font-weight:bold">${c['target']}</td>
          <td style="padding:12px 8px;text-align:center;color:{rr_color};font-weight:bold">{c['rr']}:1</td>
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
    <div style="font-size:13px;color:#ccc">{now} · S&P 500 · pre-breakout profile</div>
    <div style="margin-top:12px;font-size:12px;color:#aaa;line-height:1.6">
      Entry at current price · Stop −10% · Target +30% · Time stop 30 trading days
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
        <th style="padding:10px 8px;text-align:center">6M RS</th>
        <th style="padding:10px 8px;text-align:center">MA Spread</th>
        <th style="padding:10px 8px;text-align:center">ATR Cmp</th>
        <th style="padding:10px 8px;text-align:center">RSI</th>
        <th style="padding:10px 8px;text-align:center">4wk Ret</th>
        <th style="padding:10px 8px;text-align:center">Stop</th>
        <th style="padding:10px 8px;text-align:center">Target</th>
        <th style="padding:10px 8px;text-align:center">R:R</th>
        <th style="padding:10px 8px;text-align:center">Time Stop</th>
      </tr>
      {rows}
    </table>
    </div>

    <div style="margin-top:20px;padding:12px;background:#f9f9f9;border-radius:6px;font-size:11px;color:#888;line-height:1.6">
      ⚠️ Not investment advice. Always confirm entry with your own analysis before trading.
      Stop losses are suggested levels — execute them without hesitation.
      Score = 50MA/200MA spread (25%) · 6M RS (20%) · 4wk quietness (20%) · ATR compression (15%) · RSI zone (10%) · 1Y RS (10%).
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

    # Fetch SPY return windows as RS baseline
    print('\n[1] Fetching SPY baseline...')
    try:
        spy_hist = yf.Ticker('SPY').history(period='2y')
        spy_6m   = float(spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-126] - 1) if len(spy_hist) >= 126 else 0
        spy_1y   = float(spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-252] - 1) if len(spy_hist) >= 252 else 0
        spy_returns = {'6m': spy_6m, '1y': spy_1y}
        print(f'    SPY 6M: {round(spy_6m*100,1)}%  1Y: {round(spy_1y*100,1)}%')
    except Exception as e:
        print(f'    SPY fetch failed: {e}')
        spy_returns = {'6m': 0.0, '1y': 0.0}

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

    print('\n  TOP CANDIDATES (pre-breakout profile):')
    for c in top:
        print(f"    {c['ticker']:6} | Score {c['score']:5} | "
              f"6M RS {c.get('rs_6m',0):+.1f}% | "
              f"MA spread {c.get('ma_spread',0):+.1f}% | "
              f"RSI {c['rsi']} | ATR cmp {c.get('atr_compression',1):.2f} | "
              f"Stop ${c['stop']} | Target ${c['target']} | {c.get('sector','')}")

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

if __name__ == '__main__':
    main()
