import yfinance as yf
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────
BB_PERIOD        = 20
BB_STD           = 2
RSI_PERIOD       = 14
RSI_OVERSOLD     = 40
RSI_OVERBOUGHT   = 65
MIN_VOLUME       = 1_000_000
MIN_PRICE        = 15
MAX_PRICE        = 300
MAX_BETA         = 3.0
MAX_PE_CALLS     = 20      # PE filter for call setups
MAX_DEBT_EQUITY  = 1.0     # ratio (yfinance reports x100, we convert)
EARNINGS_BUFFER  = 21      # exclude if earnings within this many days
VOL_SPIKE_MULT   = 2.0     # volume spike = X times the 20-day average
MIN_OPTIONS_OI   = 50      # minimum open interest at ATM strike
MAX_SPREAD_PCT   = 0.30    # max bid/ask spread as % of midpoint (during market hours)
OPT_MIN_DAYS     = 14      # nearest expiration to consider (days out)
OPT_MAX_DAYS     = 45      # furthest expiration to consider (days out)
# ─────────────────────────────────────────────────────────────

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

# ── Indicator Functions ──────────────────────────────────────

def calc_rsi(prices, period=RSI_PERIOD):
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))

def calc_bb(prices):
    sma   = prices.rolling(BB_PERIOD).mean()
    sigma = prices.rolling(BB_PERIOD).std()
    return sma + BB_STD * sigma, sma, sma - BB_STD * sigma

def calc_macd(prices, fast=12, slow=26, signal=9):
    macd = prices.ewm(span=fast, adjust=False).mean() - prices.ewm(span=slow, adjust=False).mean()
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def calc_atr(h, l, c, period=14):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def format_mcap(mc):
    if mc is None or (isinstance(mc, float) and np.isnan(mc)): return 'N/A'
    if mc >= 1e12: return f'${mc/1e12:.1f}T'
    if mc >= 1e9:  return f'${mc/1e9:.1f}B'
    if mc >= 1e6:  return f'${mc/1e6:.1f}M'
    return str(mc)

# ── Options Liquidity Check ──────────────────────────────────

def check_options_liquidity(ticker, price, setup_type):
    """
    Returns (liquid: bool, atm_oi: int, spread_pct: float | None).
    Finds the nearest monthly expiration 14-45 days out, locates the
    ATM strike, and checks open interest + bid/ask spread.
    Spread check is skipped pre-market (bid=0) — OI alone decides.
    """
    try:
        t    = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return False, 0, None

        now = datetime.now()
        valid = [
            e for e in exps
            if OPT_MIN_DAYS <= (datetime.strptime(e, '%Y-%m-%d') - now).days <= OPT_MAX_DAYS
        ]
        if not valid:
            return False, 0, None

        chain   = t.option_chain(valid[0])
        options = chain.puts if 'OVERBOUGHT' in setup_type else chain.calls
        if options.empty:
            return False, 0, None

        options = options.copy()
        options['_dist'] = (options['strike'] - price).abs()
        atm = options.loc[options['_dist'].idxmin()]

        oi  = int(atm.get('openInterest', 0) or 0)
        bid = float(atm.get('bid', 0) or 0)
        ask = float(atm.get('ask', 0) or 0)

        spread_pct = None
        if bid > 0 and ask > 0:
            spread_pct = round((ask - bid) / ((ask + bid) / 2), 3)

        if oi < MIN_OPTIONS_OI:
            return False, oi, spread_pct
        if spread_pct is not None and spread_pct > MAX_SPREAD_PCT:
            return False, oi, spread_pct

        return True, oi, spread_pct
    except Exception:
        return False, 0, None

# ── PASS 1: Technical Screen (fast — history only) ───────────

def tech_screen(ticker):
    """Returns candidate dict if passes technical filters, else None."""
    try:
        hist = yf.Ticker(ticker).history(period='1y')
        if hist.empty or len(hist) < 60:
            return None

        c, h, l, v = hist['Close'], hist['High'], hist['Low'], hist['Volume']

        rsi              = calc_rsi(c)
        upper, mid, lower = calc_bb(c)
        macd, macd_sig   = calc_macd(c)
        atr              = calc_atr(h, l, c)

        price   = float(c.iloc[-1])
        cur_rsi = float(rsi.iloc[-1])
        avg_vol = float(v.tail(20).mean())
        cur_vol = float(v.iloc[-1])
        cur_up  = float(upper.iloc[-1])
        cur_mid = float(mid.iloc[-1])
        cur_low = float(lower.iloc[-1])

        # Basic filters
        if avg_vol < MIN_VOLUME: return None
        if not (MIN_PRICE <= price <= MAX_PRICE): return None
        if np.isnan(cur_rsi) or np.isnan(cur_up): return None

        # 52-week stats
        h52 = float(c.tail(252).max())
        l52 = float(c.tail(252).min())
        pct_high = round((price - h52) / h52 * 100, 1)
        pct_low  = round((price - l52) / l52  * 100, 1)

        # Days since 52w low
        days_since_low = (c.index[-1] - c.tail(252).idxmin()).days

        # SMAs
        sma50  = float(c.tail(50).mean())  if len(c) >= 50  else None
        sma200 = float(c.tail(200).mean()) if len(c) >= 200 else None

        # MACD crossover
        macd_bull = float(macd.iloc[-2]) < float(macd_sig.iloc[-2]) and float(macd.iloc[-1]) > float(macd_sig.iloc[-1])
        macd_bear = float(macd.iloc[-2]) > float(macd_sig.iloc[-2]) and float(macd.iloc[-1]) < float(macd_sig.iloc[-1])

        # Volume
        vol_spike = float(v.tail(3).max()) > avg_vol * VOL_SPIKE_MULT
        vol_ratio = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 1.0

        # Band metrics
        bw            = cur_up - cur_low
        band_pos      = round((price - cur_low) / bw * 100, 1) if bw > 0 else 50
        band_width_pct = round(bw / cur_mid * 100, 1) if cur_mid > 0 else 0
        dist_mid      = round((price - cur_mid) / cur_mid * 100, 1)

        monthly = round((price - float(c.iloc[-21])) / float(c.iloc[-21]) * 100, 1) if len(c) >= 21 else 0
        weekly  = round((price - float(c.iloc[-5]))  / float(c.iloc[-5])  * 100, 1) if len(c) >= 5  else 0

        near_lower = price <= cur_low * 1.03
        near_upper = price >= cur_up  * 0.97

        if   cur_rsi < RSI_OVERSOLD  and near_lower and pct_low  < 15:
            setup = 'SQUEEZE — Buy Call'
        elif cur_rsi > RSI_OVERBOUGHT and near_upper and pct_high > -10:
            setup = 'OVERBOUGHT — Buy Put'
        else:
            return None

        return {
            'ticker': ticker, 'setup': setup,
            'price': round(price, 2), 'cur_rsi': round(cur_rsi, 1),
            'cur_up': round(cur_up, 2), 'cur_mid': round(cur_mid, 2),
            'cur_low': round(cur_low, 2), 'band_pos': band_pos,
            'band_width_pct': band_width_pct, 'dist_mid': dist_mid,
            'pct_high': pct_high, 'pct_low': pct_low,
            'h52': round(h52, 2), 'l52': round(l52, 2),
            'days_since_low': days_since_low,
            'sma50': round(sma50, 2) if sma50 else None,
            'sma200': round(sma200, 2) if sma200 else None,
            'above_200': price > sma200 if sma200 else None,
            'macd_bull': macd_bull, 'macd_bear': macd_bear,
            'vol_spike': vol_spike, 'vol_ratio': vol_ratio,
            'avg_vol': int(avg_vol),
            'atr': round(float(atr.iloc[-1]), 2),
            'monthly': monthly, 'weekly': weekly,
        }
    except Exception:
        return None

# ── PASS 2: Fundamental Screen (candidates only) ─────────────

def fund_screen(cand):
    """Adds fundamentals, applies filters, returns final row or None."""
    ticker = cand['ticker']
    setup  = cand['setup']
    try:
        info = yf.Ticker(ticker).info

        beta    = info.get('beta',                   None)
        fwd_pe  = info.get('forwardPE',              None)
        fwd_eps = info.get('forwardEps',             None)
        eps_gr  = info.get('earningsGrowth',         None)
        de_raw  = info.get('debtToEquity',           None)
        rec     = info.get('recommendationMean',     None)
        inst    = info.get('institutionPercentHeld', None)
        mcap    = info.get('marketCap',              None)
        sector  = info.get('sector',                 'N/A')
        name    = info.get('shortName',              ticker)

        # Earnings date check
        earn_ts = info.get('earningsTimestampStart', None) or info.get('earningsTimestamp', None)
        if earn_ts:
            earn_dt    = datetime.fromtimestamp(earn_ts)
            days_until = (earn_dt - datetime.now()).days
            if 0 <= days_until <= EARNINGS_BUFFER:
                return None  # too close to earnings

        debt_equity = round(de_raw / 100, 2) if de_raw is not None else None

        # ── Risk Filters ──────────────────────────────────
        if beta is not None and beta > MAX_BETA:
            return None

        # Call-specific filters
        if 'SQUEEZE' in setup:
            if fwd_pe  is not None and fwd_pe > MAX_PE_CALLS:  return None
            if fwd_eps is not None and fwd_eps <= 0:            return None
            if debt_equity is not None and debt_equity > MAX_DEBT_EQUITY: return None
            if rec  is not None and rec > 4.0:                  return None  # strong sell only

        # ── Options Liquidity Check ───────────────────────────
        opt_liquid, opt_oi, opt_spread = check_options_liquidity(ticker, cand['price'], setup)
        if not opt_liquid:
            return None

        # ── Signal Strength Score (0–100) ─────────────────
        s = cand
        score = 0.0

        if 'SQUEEZE' in setup:
            score += min(25, max(0, (RSI_OVERSOLD - s['cur_rsi'])  * 1.5))
            score += min(25, max(0,  25 - s['pct_low']             * 1.5))
            score += min(20, s['vol_ratio'] * 7)
            score += min(15, abs(min(0, s['weekly']))               * 1.5)
            score += min(15, abs(min(0, s['dist_mid']))             * 1.5)
            if s['macd_bull']:           score += 5
            if s['vol_spike']:           score += 5
            if s['above_200']:           score += 3
            if s['days_since_low'] <= 5: score += 3
            if rec is not None and rec <= 2.0: score += 4   # analysts bullish
        else:
            score += min(25, max(0, (s['cur_rsi'] - RSI_OVERBOUGHT) * 1.2))
            score += min(25, max(0,  25 + s['pct_high']             * 2.0))
            score += min(20, s['vol_ratio'] * 7)
            score += min(15, max(0,  s['weekly'])                   * 1.5)
            score += min(15, max(0,  s['dist_mid'])                 * 1.5)
            if s['macd_bear']:                          score += 5
            if s['vol_spike']:                          score += 5
            if s['above_200'] is False:                 score += 3
            if fwd_pe is not None and fwd_pe > 30:      score += 4   # overvalued bonus

        return {
            'Ticker':           ticker,
            'Company':          name,
            'Sector':           sector,
            'Setup':            setup,
            'Signal Strength':  round(min(100, score), 1),
            # ── Price & Bands ──
            'Price':            s['price'],
            'RSI':              s['cur_rsi'],
            'Lower BB':         s['cur_low'],
            'Middle BB':        s['cur_mid'],
            'Upper BB':         s['cur_up'],
            'Band Position %':  s['band_pos'],
            'Dist from Mid %':  s['dist_mid'],
            'Band Width %':     s['band_width_pct'],
            # ── 52-Week ──
            '52W Low':          s['l52'],
            '52W High':         s['h52'],
            '% From 52W Low':   s['pct_low'],
            '% From 52W High':  s['pct_high'],
            'Days Since 52W Low': s['days_since_low'] if 'SQUEEZE' in setup else '—',
            # ── Trend ──
            'SMA 50':           s['sma50'] if s['sma50'] else 'N/A',
            'SMA 200':          s['sma200'] if s['sma200'] else 'N/A',
            'Above 200 SMA':    'Yes' if s['above_200'] else 'No',
            # ── Momentum ──
            'MACD Cross':       ('Bullish ✓' if s['macd_bull'] else ('Bearish ✓' if s['macd_bear'] else '—')),
            'Vol Spike':        'Yes ✓' if s['vol_spike'] else '—',
            'Vol Ratio (1d)':   s['vol_ratio'],
            'Avg Volume':       f"{s['avg_vol']:,}",
            'ATR':              s['atr'],
            'Monthly %':        s['monthly'],
            'Weekly %':         s['weekly'],
            # ── Fundamentals ──
            'Forward PE':       round(fwd_pe, 1) if fwd_pe else 'N/A',
            'Forward EPS':      round(fwd_eps, 2) if fwd_eps else 'N/A',
            'EPS Growth':       f"{round(eps_gr*100,1)}%" if eps_gr else 'N/A',
            'Beta':             round(beta, 2) if beta else 'N/A',
            'Debt/Equity':      debt_equity if debt_equity is not None else 'N/A',
            'Analyst Rating':   round(rec, 1) if rec else 'N/A',
            'Inst. Ownership':  f"{round(inst*100,1)}%" if inst else 'N/A',
            'Market Cap':       format_mcap(mcap),
            # ── Options Liquidity ──
            'ATM OI':           opt_oi,
            'Spread %':         f"{round(opt_spread*100,1)}%" if opt_spread is not None else 'pre-mkt',
        }
    except Exception:
        return None

# ── Excel Writer ─────────────────────────────────────────────

COLUMNS = [
    'Ticker','Company','Sector','Setup','Signal Strength',
    'Price','RSI','Lower BB','Middle BB','Upper BB','Band Position %','Dist from Mid %','Band Width %',
    '52W Low','52W High','% From 52W Low','% From 52W High','Days Since 52W Low',
    'SMA 50','SMA 200','Above 200 SMA',
    'MACD Cross','Vol Spike','Vol Ratio (1d)','Avg Volume','ATR','Monthly %','Weekly %',
    'Forward PE','Forward EPS','EPS Growth','Beta','Debt/Equity','Analyst Rating','Inst. Ownership','Market Cap',
    'ATM OI','Spread %',
]
COL_WIDTHS = [
    8,24,18,22,14,
    8,6,10,10,10,14,14,12,
    10,10,14,15,16,
    10,10,13,
    12,10,14,14,8,10,10,
    11,12,11,7,12,13,14,12,
    10,10,
]

def write_sheet(wb, data, sheet_name, title, hdr_color, row_color):
    ws = wb.create_sheet(sheet_name)
    hf = PatternFill(start_color=hdr_color, end_color=hdr_color, fill_type='solid')
    rf = PatternFill(start_color=row_color,  end_color=row_color,  fill_type='solid')
    wb_font = Font(color='FFFFFF', bold=True)
    ctr     = Alignment(horizontal='center')

    ws.append([title])
    ws['A1'].font = Font(bold=True, size=14)
    ws['A1'].alignment = ctr
    ws.merge_cells(f'A1:{get_column_letter(len(COLUMNS))}1')

    ws.append([f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}  |  {len(data)} results'])
    ws['A2'].alignment = ctr
    ws.merge_cells(f'A2:{get_column_letter(len(COLUMNS))}2')
    ws.append([])

    ws.append(COLUMNS)
    for cell in ws[ws.max_row]:
        cell.fill = hf; cell.font = wb_font; cell.alignment = ctr

    for r in sorted(data, key=lambda x: x['Signal Strength'], reverse=True):
        ws.append([r.get(c, '') for c in COLUMNS])
        for cell in ws[ws.max_row]:
            cell.fill = rf; cell.alignment = ctr

    for i, w in enumerate(COL_WIDTHS, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

# ── Main ─────────────────────────────────────────────────────

def main():
    print('=' * 58)
    print('  BOLLINGER BAND SCREENER v3 — S&P 500')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 58)

    tickers = SP500
    print(f'\n[Pass 1] Technical scan — {len(tickers)} tickers...\n')

    candidates = []
    for i, t in enumerate(tickers):
        if i % 50 == 0:
            print(f'  [{i:>3}/{len(tickers)}]  Candidates so far: {len(candidates)}')
        r = tech_screen(t)
        if r:
            candidates.append(r)

    squeeze_c    = sum(1 for c in candidates if 'SQUEEZE'    in c['setup'])
    overbought_c = sum(1 for c in candidates if 'OVERBOUGHT' in c['setup'])
    print(f'\n  Technical candidates: {len(candidates)}  '
          f'(Call: {squeeze_c}  |  Put: {overbought_c})')

    print(f'\n[Pass 2] Fundamental screen + options liquidity + scoring — {len(candidates)} candidates...\n')

    results = []
    for i, cand in enumerate(candidates):
        if i % 10 == 0:
            print(f'  [{i:>2}/{len(candidates)}]  Passed so far: {len(results)}')
        r = fund_screen(cand)
        if r:
            results.append(r)

    squeeze_r    = [r for r in results if 'SQUEEZE'    in r['Setup']]
    overbought_r = [r for r in results if 'OVERBOUGHT' in r['Setup']]

    print(f'\n{"=" * 58}')
    print(f'  FINAL RESULTS')
    print(f'  Call setups (Squeeze):    {len(squeeze_r)}')
    print(f'  Put setups (Overbought):  {len(overbought_r)}')
    print(f'  Total high-conviction:    {len(results)}')
    print(f'{"=" * 58}')

    if squeeze_r:
        top = sorted(squeeze_r, key=lambda x: x['Signal Strength'], reverse=True)[:3]
        print('\n  TOP CALL SETUPS:')
        for r in top:
            print(f'    {r["Ticker"]:6} | Score {r["Signal Strength"]:5} | RSI {r["RSI"]} | {r["Sector"]}')

    if overbought_r:
        top = sorted(overbought_r, key=lambda x: x['Signal Strength'], reverse=True)[:3]
        print('\n  TOP PUT SETUPS:')
        for r in top:
            print(f'    {r["Ticker"]:6} | Score {r["Signal Strength"]:5} | RSI {r["RSI"]} | {r["Sector"]}')

    if not results:
        print('\n  No high-conviction setups found today.')
        return

    wb = Workbook()
    wb.remove(wb.active)

    write_sheet(wb, squeeze_r,
                'Call Setups — Squeeze',
                'SQUEEZE / BOUNCE  (Buy Calls)',
                '1A4D1A', 'D6F5D6')

    write_sheet(wb, overbought_r,
                'Put Setups — Overbought',
                'OVERBOUGHT  (Buy Puts / Inverse ETF)',
                '7B0000', 'FFD9D9')

    fname = f'/Users/colethompson/Downloads/BB_Screener_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx'
    wb.save(fname)
    print(f'\n  Saved → {fname}')

if __name__ == '__main__':
    main()
