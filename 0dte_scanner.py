#!/usr/bin/env python3
"""
0dte_scanner.py — 0DTE SPY options scanner

Analyzes market conditions across 7 signal layers and recommends
a SPY 0DTE call or put setup with specific strikes from the live chain.

Usage:
  python3 ~/Downloads/0dte_scanner.py            # print to terminal
  python3 ~/Downloads/0dte_scanner.py --email    # print + email results
"""

import os, sys, ssl, smtplib, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from urllib.parse import quote
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import yfinance as yf
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# ── CONFIG ────────────────────────────────────────────────────────────────
EMAIL_TO           = "cole.w.thompson.01@gmail.com"
EMAIL_FROM         = "cole.w.thompson.01@gmail.com"
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
SEND_EMAIL         = "--email" in sys.argv

MEGA_CAPS = ["NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "TSLA"]

# Tickers to scan — only those with today's 0DTE expiry will be included
SCAN_TICKERS = ["NVDA", "TSLA", "META", "AMZN", "MU", "NFLX", "MSTR", "HOOD",
                "QQQ", "PLTR", "CRWD", "ARM", "COIN", "SMCI"]
# Removed (sub-50% under super-filter): AMD 33%, GOOGL 33%, PANW 43%
# Removed (low sample / poor historical): SPY, IWM, AAPL, MSFT

SECTORS = {
    "Tech":       "XLK",
    "Financials": "XLF",
    "Energy":     "XLE",
    "Healthcare": "XLV",
    "Comms":      "XLC",
    "Industrials":"XLI",
}

# Sector ETF benchmark per ticker (for relative-strength signal)
TICKER_SECTOR_ETF = {
    "NVDA": "SMH",
    "AMD":  "SMH",
    "MU":   "SMH",
    "TSLA": "XLY",
    "META": "XLC",
    "AMZN": "XLC",
    "NFLX": "XLC",
    "MSTR": "XLF",
    "HOOD": "XLF",
}

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE


# ── HELPERS ───────────────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return round(float((100 - 100 / (1 + rs)).iloc[-1]), 1)

def calc_bb_pctb(series, period=20):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return round(float(((series - lower) / (upper - lower)).iloc[-1]), 2)

def first_friday(d=None):
    d = d or date.today()
    first = d.replace(day=1)
    return first + timedelta(days=(4 - first.weekday()) % 7)

def rss_fetch(query, max_items=8):
    items = []
    try:
        url = (f"https://news.google.com/rss/search"
               f"?q={quote(query)}&hl=en-US&gl=US&ceid=US:en")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8, context=_SSL_CTX) as r:
            root = ET.fromstring(r.read())
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            if " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()
            if title:
                items.append(title)
            if len(items) >= max_items:
                break
    except Exception:
        pass
    return items

# ── 1. CATALYSTS ──────────────────────────────────────────────────────────
def get_catalysts():
    today     = date.today()
    catalysts = []
    score     = 0

    if today == first_friday(today):
        catalysts.append("Non-Farm Payrolls (Jobs Report)")
        score += 3

    for ticker in MEGA_CAPS:
        try:
            cal = yf.Ticker(ticker).calendar
            if not cal:
                continue
            dates = cal.get("Earnings Date", []) if isinstance(cal, dict) else []
            if not isinstance(dates, list):
                dates = [dates]
            for ed in dates:
                ed_date = ed.date() if hasattr(ed, "date") else ed
                if isinstance(ed_date, date) and ed_date == today:
                    catalysts.append(f"{ticker} earnings today")
                    score += 2
        except Exception:
            pass

    headlines = rss_fetch("stock market today economic data federal reserve", max_items=12)
    text = " ".join(headlines).lower()
    macro_events = [
        (["fomc", "fed decision", "rate decision", "federal reserve decision"], "Fed rate decision / FOMC", 3),
        (["cpi", "consumer price index", "inflation report"],                   "CPI inflation data",       3),
        (["nonfarm payroll", "jobs report", "unemployment rate"],               "Jobs / unemployment data", 3),
        (["pce", "personal consumption expenditure"],                           "PCE inflation data",       2),
        (["gdp", "gross domestic product"],                                     "GDP release",              2),
        (["ppi", "producer price"],                                             "PPI data",                 2),
        (["retail sales"],                                                      "Retail sales data",        1),
    ]
    seen = set()
    for keywords, label, pts in macro_events:
        if any(kw in text for kw in keywords) and label not in seen:
            catalysts.append(label)
            seen.add(label)
            score += pts

    return catalysts, min(score, 5)

# ── 2. MARKET TECHNICALS ─────────────────────────────────────────────────
def get_market_technicals():
    result = {}

    try:
        spy    = yf.Ticker("SPY")
        hist   = spy.history(period="30d")
        hist1m = spy.history(period="1d", interval="1m", prepost=True)
        hist2d = spy.history(period="2d")

        prev_close            = float(hist2d["Close"].iloc[-2])
        current               = float(hist1m["Close"].iloc[-1])
        result["spy_price"]   = round(current, 2)
        result["spy_prev"]    = round(prev_close, 2)
        result["spy_premkt"]  = round((current / prev_close - 1) * 100, 2)
        result["spy_rsi"]     = calc_rsi(hist["Close"])
        result["spy_pctb"]    = calc_bb_pctb(hist["Close"])
    except Exception as e:
        result["spy_error"] = str(e)

    try:
        es   = yf.Ticker("ES=F").history(period="2d")
        result["futures_price"] = round(float(es["Close"].iloc[-1]), 1)
        result["futures_pct"]   = round((float(es["Close"].iloc[-1]) /
                                         float(es["Close"].iloc[-2]) - 1) * 100, 2)
    except Exception:
        pass

    try:
        vix_hist          = yf.Ticker("^VIX").history(period="1y")
        result["vix"]     = round(float(vix_hist["Close"].iloc[-1]), 2)
        result["vix_prev"]= round(float(vix_hist["Close"].iloc[-2]), 2)
        result["vix_chg"] = round(result["vix"] - result["vix_prev"], 2)
        # IVR: where current VIX sits in its 52-week range (0–100)
        lo, hi = float(vix_hist["Close"].min()), float(vix_hist["Close"].max())
        if hi > lo:
            result["vix_ivr"] = round((result["vix"] - lo) / (hi - lo) * 100, 1)
    except Exception:
        pass

    # VIX term structure: VIX / VIX3M — >1.0 = backwardation (fear premium)
    try:
        vix3m = yf.Ticker("^VIX3M").history(period="2d")
        if len(vix3m) >= 1 and "vix" in result:
            result["vix_ts"] = round(result["vix"] / float(vix3m["Close"].iloc[-1]), 3)
    except Exception:
        pass

    return result

# ── 2b. PUT/CALL RATIO ───────────────────────────────────────────────────
def get_put_call_ratio():
    """SPY market put/call ratio from nearest-expiry options chain."""
    try:
        spy  = yf.Ticker("SPY")
        exps = spy.options
        if not exps:
            return None
        chain    = spy.option_chain(exps[0])
        put_vol  = chain.puts["volume"].fillna(0).sum()
        call_vol = chain.calls["volume"].fillna(0).sum()
        if call_vol > 0:
            return round(float(put_vol / call_vol), 2)
    except Exception:
        pass
    return None

# ── 3. CROSS-ASSET ────────────────────────────────────────────────────────
def get_cross_asset():
    tickers = {
        "10Y Yield": "^TNX",
        "DXY":       "DX-Y.NYB",
        "Oil (WTI)": "CL=F",
        "Gold":      "GC=F",
        "Bitcoin":   "BTC-USD",
    }
    result = {}
    for name, sym in tickers.items():
        try:
            h = yf.Ticker(sym).history(period="2d")
            if len(h) >= 2:
                prev = float(h["Close"].iloc[-2])
                last = float(h["Close"].iloc[-1])
                result[name] = {"price": round(last, 2),
                                "pct":   round((last / prev - 1) * 100, 2)}
        except Exception:
            pass
    return result

# ── 4. SECTOR PULSE ───────────────────────────────────────────────────────
def get_sector_pulse():
    result = {}
    for name, sym in SECTORS.items():
        try:
            h = yf.Ticker(sym).history(period="2d")
            if len(h) >= 2:
                result[name] = round((float(h["Close"].iloc[-1]) /
                                      float(h["Close"].iloc[-2]) - 1) * 100, 2)
        except Exception:
            pass
    return result

# ── 5. NEWS SENTIMENT ─────────────────────────────────────────────────────
def get_news_sentiment():
    headlines = rss_fetch("S&P 500 stock market today", max_items=12)
    scored = []
    compounds = []
    for h in headlines:
        c = _vader.polarity_scores(h)["compound"]
        compounds.append(c)
        tone = "+" if c >= 0.05 else ("-" if c <= -0.05 else "~")
        scored.append((tone, h))
    avg = sum(compounds) / len(compounds) if compounds else 0.0
    label = "BULLISH" if avg >= 0.05 else ("BEARISH" if avg <= -0.05 else "NEUTRAL")
    return label, round(avg, 3), scored[:6]

# ── 6. ANALYST SIGNALS ───────────────────────────────────────────────────
def get_analyst_signals():
    """Pull recent upgrades/downgrades for mega-caps — net count as market signal."""
    upgrades = downgrades = 0
    details  = []
    cutoff   = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    upgrade_actions   = {"upgrade", "upgraded to", "initiated", "reiterated", "raised"}
    downgrade_actions = {"downgrade", "downgraded to", "lowered", "cut", "reduced"}

    today_utc = pd.Timestamp.now(tz="UTC").normalize()
    for ticker in MEGA_CAPS:
        try:
            df = yf.Ticker(ticker).upgrades_downgrades
            if df is None or df.empty:
                continue
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df = df[df.index >= cutoff]
            for ts, row in df.iterrows():
                action = str(row.get("Action", row.get("ToGrade", ""))).lower()
                firm   = str(row.get("Firm", ""))
                grade  = str(row.get("ToGrade", ""))
                fresh  = ts >= today_utc
                badge  = " 🔴 FRESH TODAY" if fresh else ""
                is_up  = any(w in action for w in upgrade_actions)
                is_dn  = any(w in action for w in downgrade_actions)
                if is_up:
                    upgrades += 1
                    details.append(f"↑ {ticker} — {firm} ({grade}){badge}")
                elif is_dn:
                    downgrades += 1
                    details.append(f"↓ {ticker} — {firm} ({grade}){badge}")
        except Exception:
            pass

    net   = upgrades - downgrades
    label = "BULLISH" if net > 0 else ("BEARISH" if net < 0 else "NEUTRAL")
    return upgrades, downgrades, net, label, details[:6]

# ── 7. SPY 0DTE CHAIN ─────────────────────────────────────────────────────
def get_spy_0dte_chain(spy_price, direction):
    try:
        spy    = yf.Ticker("SPY")
        today  = date.today().strftime("%Y-%m-%d")
        exps   = spy.options
        exp    = today if today in exps else (exps[0] if exps else None)
        if not exp:
            return [], None

        chain = spy.option_chain(exp)
        df    = (chain.calls if direction == "CALL" else chain.puts).copy()

        # Near ATM ± 2%
        df = df[(df["strike"] >= spy_price * 0.98) &
                (df["strike"] <= spy_price * 1.02)]

        # Slight OTM bias for each direction
        if direction == "CALL":
            df = df[df["strike"] >= spy_price * 0.998]
        else:
            df = df[df["strike"] <= spy_price * 1.002]

        df = df[df["openInterest"].fillna(0) > 10]
        df["_dist"] = abs(df["strike"] - spy_price)
        df = df.sort_values("_dist")

        keep = ["strike", "lastPrice", "bid", "ask", "volume",
                "openInterest", "impliedVolatility", "delta"]
        cols = [c for c in keep if c in df.columns]
        return df[cols].head(4).to_dict("records"), exp
    except Exception:
        return [], None

# ── 7b. 0DTE EXPIRY CHECK ────────────────────────────────────────────────
def check_0dte_expiry(ticker):
    """Return today's expiry string if 0DTE options exist for this ticker, else None."""
    try:
        today = date.today().strftime("%Y-%m-%d")
        exps  = yf.Ticker(ticker).options
        return today if today in exps else None
    except Exception:
        return None

# ── 7c. PER-TICKER TECHNICALS ────────────────────────────────────────────
def get_ticker_technicals(ticker, market_tech):
    """Fetch RSI, BB%B, pre-market move for any ticker; merge market-wide signals."""
    result = {
        "futures_pct":    market_tech.get("futures_pct", 0),
        "vix":            market_tech.get("vix"),
        "vix_chg":        market_tech.get("vix_chg", 0),
        "vix_ivr":        market_tech.get("vix_ivr"),
        "vix_ts":         market_tech.get("vix_ts"),
        "put_call_ratio": market_tech.get("put_call_ratio"),
    }
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="60d")   # 60d needed for 50-MA trend regime
        h1m  = t.history(period="1d", interval="1m", prepost=True)
        h2d  = t.history(period="2d")
        prev    = float(h2d["Close"].iloc[-2])
        current = float(h1m["Close"].iloc[-1])
        result["tk_price"]  = round(current, 2)
        result["tk_premkt"] = round((current / prev - 1) * 100, 2)
        result["tk_rsi"]    = calc_rsi(hist["Close"])
        result["tk_pctb"]   = calc_bb_pctb(hist["Close"])

        # Gap ATR: today's gap size relative to ATR(14)
        try:
            today_open = float(h1m["Open"].iloc[0])
            tr = pd.concat([
                hist["High"] - hist["Low"],
                (hist["High"] - hist["Close"].shift(1)).abs(),
                (hist["Low"]  - hist["Close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-2])
            if atr > 0:
                result["gap_atr"] = round((today_open - prev) / atr, 2)
        except Exception:
            pass

        # Trend regime: price vs 20/50 MA alignment
        try:
            s20 = float(hist["Close"].rolling(20).mean().iloc[-1])
            s50 = float(hist["Close"].rolling(50).mean().iloc[-1])
            if prev > s20 and s20 > s50:
                result["trend_regime"] = 1
            elif prev < s20 and s20 < s50:
                result["trend_regime"] = -1
            else:
                result["trend_regime"] = 0
        except Exception:
            pass

        # Relative strength vs sector ETF (5-day)
        try:
            etf_sym = TICKER_SECTOR_ETF.get(ticker)
            if etf_sym and len(hist) >= 6:
                etf_hist = yf.Ticker(etf_sym).history(period="15d")
                if len(etf_hist) >= 6:
                    etf_rs = float(etf_hist["Close"].iloc[-1]) / float(etf_hist["Close"].iloc[-6]) - 1
                    tk_rs  = float(hist["Close"].iloc[-1])     / float(hist["Close"].iloc[-6])     - 1
                    result["rel_strength"] = round((tk_rs - etf_rs) * 100, 2)
        except Exception:
            pass

        # Intraday signals via Alpaca minute bars (optional — needs ALPACA_API_KEY)
        try:
            if os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"):
                import importlib.util as _ilu
                _sp = os.path.join(os.path.dirname(__file__), "alpaca_data.py")
                if os.path.exists(_sp):
                    _spec = _ilu.spec_from_file_location("alpaca_data", _sp)
                    _amod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_amod)
                    # Fetch 25 days so RVOL has 20-session history
                    _start = date.today() - timedelta(days=25)
                    _bars  = _amod.fetch_minute_bars(ticker, _start,
                                                     date.today() + timedelta(days=1))
                    if not _bars.empty:
                        _atr_val = result.get("gap_atr")
                        # Pre-compute RVOL for all days in the 25-day window
                        _rvol_lookup = _amod.precompute_rvol(_bars)
                        _sig = _amod.build_intraday_signals(ticker, _bars, date.today(),
                                                            atr14=_atr_val,
                                                            rvol_lookup=_rvol_lookup)
                        if _sig:
                            result["entry_vs_vwap"] = _sig["entry_vs_vwap"]
                            result["or_range_pct"]  = _sig["or_range_pct"]
                            result["orb_direction"] = _sig["orb_direction"]
                            result["rvol"]          = _sig["rvol"]
                            result["or_atr"]        = _sig["or_atr"]
                            result["gap_sig"]       = _sig["gap_sig"]
        except Exception:
            pass

        # Earnings veto: skip ±1 trading day of earnings
        try:
            cal = yf.Ticker(ticker).calendar
            if cal:
                dates = cal.get("Earnings Date", []) if isinstance(cal, dict) else []
                if not isinstance(dates, list):
                    dates = [dates]
                today = date.today()
                for ed in dates:
                    ed_date = ed.date() if hasattr(ed, "date") else ed
                    if isinstance(ed_date, date) and abs((ed_date - today).days) <= 1:
                        result["earnings_veto"] = True
                        break
        except Exception:
            pass

    except Exception as e:
        result["tk_error"] = str(e)
    return result

# ── 7d. PER-TICKER NEWS ───────────────────────────────────────────────────
def get_ticker_news(ticker):
    """VADER sentiment for a specific ticker's RSS headlines."""
    headlines = rss_fetch(f"{ticker} stock today", max_items=8)
    if not headlines:
        return "NEUTRAL", 0.0
    compounds = [_vader.polarity_scores(h)["compound"] for h in headlines]
    avg   = sum(compounds) / len(compounds)
    label = "BULLISH" if avg >= 0.05 else ("BEARISH" if avg <= -0.05 else "NEUTRAL")
    return label, round(avg, 3)

# ── 7e. GENERALIZED 0DTE CHAIN ───────────────────────────────────────────
def get_0dte_chain(ticker, price, direction):
    try:
        t     = yf.Ticker(ticker)
        today = date.today().strftime("%Y-%m-%d")
        exps  = t.options
        exp   = today if today in exps else (exps[0] if exps else None)
        if not exp:
            return [], None
        chain = t.option_chain(exp)
        df    = (chain.calls if direction == "CALL" else chain.puts).copy()
        df    = df[(df["strike"] >= price * 0.97) & (df["strike"] <= price * 1.03)]
        if direction == "CALL":
            df = df[df["strike"] >= price * 0.998]
        else:
            df = df[df["strike"] <= price * 1.002]
        df = df[df["openInterest"].fillna(0) > 10]
        df["_dist"] = abs(df["strike"] - price)
        df = df.sort_values("_dist")
        keep = ["strike", "lastPrice", "bid", "ask", "volume",
                "openInterest", "impliedVolatility", "delta"]
        cols = [c for c in keep if c in df.columns]
        return df[cols].head(4).to_dict("records"), exp
    except Exception:
        return [], None

# ── 7f. SCAN ONE TICKER ───────────────────────────────────────────────────
def _check_fomc_day():
    """Returns True if today appears to be an FOMC decision day (live only)."""
    try:
        text = " ".join(rss_fetch("FOMC meeting today fed decision", max_items=5)).lower()
        return any(w in text for w in ["fomc meeting today", "rate decision today",
                                       "fed announces", "fed raises", "fed cuts",
                                       "federal open market committee"])
    except Exception:
        return False

_FOMC_TODAY = None  # cached once per scanner run

def scan_ticker(ticker, market_tech, cross, sectors, cat_score):
    """Run full signal analysis for one ticker. Returns result dict."""
    global _FOMC_TODAY
    if _FOMC_TODAY is None:
        _FOMC_TODAY = _check_fomc_day()

    exp_date = check_0dte_expiry(ticker)
    if not exp_date:
        return {"ticker": ticker, "has_0dte": False}

    tech               = get_ticker_technicals(ticker, market_tech)
    sentiment, sent_net = get_ticker_news(ticker)
    direction, conviction, go_nogo, signals = build_recommendation(
        tech, cross, sectors, sentiment, cat_score, fomc_day=_FOMC_TODAY
    )
    price = tech.get("tk_price", 0)
    chain, _ = get_0dte_chain(ticker, price, direction) if direction != "NO TRADE" else ([], None)

    return {
        "ticker":     ticker,
        "has_0dte":   True,
        "tech":       tech,
        "direction":  direction,
        "conviction": conviction,
        "go_nogo":    go_nogo,
        "signals":    signals,
        "chain":      chain,
        "exp_date":   exp_date,
        "sentiment":  sentiment,
        "sent_net":   sent_net,
    }

# ── 8. SCORE + RECOMMEND ──────────────────────────────────────────────────
import math as _math

# ── Signal weight tuning (Fable-optimized, Nov 2026 backtest) ────────────────
_GO_THRESHOLD_CALL    = 7.0   # min conviction for CALL → GO
_GO_THRESHOLD_PUT     = 7.0   # same threshold — PUTs now perform equally
_GO_CONVICTION_MAX    = 8.0   # cap — above 8 is over-signaled (44% win vs 60% at 7-8)
_MIN_STRENGTH         = 0.25  # min signal strength to assign a direction
_MIN_ACTIVE_SIGNALS   = 3     # min non-neutral signals to assign a direction
_MIXED_MINORITY_SHARE = 0.25  # dissenting weight share that flags mixed
_MIXED_CONVICTION_CAP = 5.0   # below both GO thresholds → mixed blocks trade
_VIX_MIN, _VIX_MAX   = 15.0, 28.0
_CAT_BASELINE         = 2.0
_CAT_SLOPE            = 0.75
_TECH_LABELS = ("S&P Futures", "Pre-Market Move", "RSI (14)", "BB %B")

def build_recommendation(tech, cross, sectors, sentiment, cat_score, analyst_net=0,
                         fomc_day=False):
    signals = []  # (label, display_value, vote [-1/0/1], weight)

    if "futures_pct" in tech:
        fp   = tech["futures_pct"]
        vote = 1 if fp > 0.1 else (-1 if fp < -0.1 else 0)
        signals.append(("S&P Futures", f"{fp:+.2f}%", vote, 1.25))   # 55.9% → up from 1.0

    premkt = tech.get("tk_premkt", tech.get("spy_premkt"))
    gap_filled = tech.get("gap_sig") == 0 and tech.get("gap_atr", 0) != 0  # stale if gap filled
    if premkt is not None:
        vote = 1 if premkt > 0.1 else (-1 if premkt < -0.1 else 0)
        pm_w = 0.0 if gap_filled else 0.75   # zero out if gap already filled (Fable rec)
        signals.append(("Pre-Market Move", f"{premkt:+.2f}%", vote, pm_w))

    if "vix_chg" in tech:                            # 57.6% — highest
        vc   = tech["vix_chg"]
        vote = 1 if vc < -0.5 else (-1 if vc > 0.5 else 0)
        signals.append(("VIX Direction", f"{vc:+.2f}", vote, 2.0))

    if "10Y Yield" in cross:                         # 57.8% — raised from 1.25
        yp   = cross["10Y Yield"]["pct"]
        vote = -1 if yp > 0.5 else (1 if yp < -0.5 else 0)
        signals.append(("10Y Yield", f"{yp:+.2f}%", vote, 1.75))

    if "DXY" in cross:                               # 52.4% — lowered from 1.75
        dp   = cross["DXY"]["pct"]
        vote = -1 if dp > 0.3 else (1 if dp < -0.3 else 0)
        signals.append(("DXY", f"{dp:+.2f}%", vote, 1.0))

    if sentiment is not None:                        # unvalidated — low weight
        sent_vote = 1 if sentiment == "BULLISH" else (-1 if sentiment == "BEARISH" else 0)
        signals.append(("News Sentiment", sentiment, sent_vote, 0.5))

    if sectors:                                      # 54.4% — lowered from 1.75
        pos = sum(1 for v in sectors.values() if v > 0)
        neg = sum(1 for v in sectors.values() if v < 0)
        sec_vote = 1 if pos > neg else (-1 if neg > pos else 0)
        signals.append(("Sector Pulse", f"{pos} up / {neg} dn", sec_vote, 1.25))

    rsi = tech.get("tk_rsi", tech.get("spy_rsi"))
    if rsi is not None:
        vote = 1 if rsi < 45 else (-1 if rsi > 60 else 0)
        signals.append(("RSI (14)", str(rsi), vote, 0.5))

    pctb = tech.get("tk_pctb", tech.get("spy_pctb"))
    if pctb is not None:
        vote = -1 if pctb > 0.8 else (1 if pctb < 0.2 else 0)
        signals.append(("BB %B", str(pctb), vote, 0.5))

    if analyst_net != 0:
        a_vote = 1 if analyst_net > 0 else -1
        signals.append(("Analyst Ratings", f"{analyst_net:+d} net (7d)", a_vote, 0.25))

    # VIX IVR — where current VIX sits in its 52-week range
    # Low IVR (<30) = cheap options, favorable to buy; High (>70) = expensive, IV crush risk
    vix_ivr = tech.get("vix_ivr")
    if vix_ivr is not None:
        ivr_vote = 1 if vix_ivr < 30 else (-1 if vix_ivr > 70 else 0)
        signals.append(("VIX IVR", f"{vix_ivr:.0f}th pctile", ivr_vote, 0.75))

    # Put/Call Ratio — market sentiment (live only, not in backtest)
    # High P/C (>1.1) = heavy put buying = bearish; Low (<0.75) = bullish complacency
    pcr = tech.get("put_call_ratio")
    if pcr is not None:
        pcr_vote = -1 if pcr > 1.1 else (1 if pcr < 0.75 else 0)
        signals.append(("Put/Call Ratio", f"{pcr:.2f}", pcr_vote, 0.75))

    # VWAP position at 10 AM CT (55.8% accuracy — raised from 1.0)
    entry_vs_vwap = tech.get("entry_vs_vwap")
    rvol = tech.get("rvol")
    if entry_vs_vwap is not None:
        vwap_vote = 1 if entry_vs_vwap > 0.1 else (-1 if entry_vs_vwap < -0.1 else 0)
        # Scale by RVOL: high volume = more conviction in VWAP signal
        rvol_mult = 1.25 if (rvol and rvol >= 1.5) else (0.75 if (rvol and rvol < 0.5) else 1.0)
        signals.append(("VWAP Position", f"{entry_vs_vwap:+.2f}% vs VWAP", vwap_vote,
                        round(1.25 * rvol_mult, 2)))

    # ORB direction: price above/below true opening range (8:30-9:00 AM CT) at entry
    orb_dir = tech.get("orb_direction")
    if orb_dir is not None and orb_dir != 0:
        or_atr = tech.get("or_atr")
        # Scale by OR width: narrow range = more ORB signal conviction
        or_mult = 1.25 if (or_atr and or_atr < 0.25) else (0.5 if (or_atr and or_atr > 1.0) else 1.0)
        orb_w   = round(0.75 * rvol_mult * or_mult, 2)
        signals.append(("ORB Direction",
                        f"{'Above' if orb_dir > 0 else 'Below'} OR" +
                        (f" (or_atr={or_atr:.2f})" if or_atr else ""),
                        orb_dir, orb_w))

    # Gap fill state: gap-and-go (+1) or failed gap (-1)
    gap_sig_val = tech.get("gap_sig")
    if gap_sig_val is not None and gap_sig_val != 0:
        signals.append(("Gap Fill State",
                        "Gap-and-go" if gap_sig_val > 0 else "Failed gap",
                        gap_sig_val, 0.5))

    # Gap ATR — gap size relative to ATR(14); >1.5 = exhaustion (neutral)
    gap_atr = tech.get("gap_atr")
    if gap_atr is not None:
        if abs(gap_atr) > 1.5:
            gap_vote, gap_lbl = 0, f"{gap_atr:.2f}x ATR (exhaustion)"
        elif abs(gap_atr) >= 0.25:
            gap_vote = 1 if gap_atr > 0 else -1
            gap_lbl  = f"{gap_atr:.2f}x ATR"
        else:
            gap_vote, gap_lbl = 0, f"{gap_atr:.2f}x ATR (small)"
        signals.append(("Gap ATR", gap_lbl, gap_vote, 0.75))

    # Trend regime — 20/50 MA alignment; only trade with the trend
    tr_regime = tech.get("trend_regime")
    if tr_regime is not None and tr_regime != 0:
        signals.append(("Trend Regime", "Bullish" if tr_regime > 0 else "Bearish",
                        tr_regime, 1.0))


    # ── VIX regime: hard gate (replaces old -1.5 penalty) ───────────────────
    vix = tech.get("vix", 20)
    vix_is_num = isinstance(vix, (int, float))
    if vix_is_num and vix < _VIX_MIN:
        regime = "LOW"
    elif vix_is_num and vix > _VIX_MAX:
        regime = "HIGH"
    else:
        regime = "NORMAL"
    vix_ok = regime == "NORMAL"

    # regime multipliers kept at 1.0 — zero out-of-regime trades in backtest
    weighted_signals = []
    for label, val, vote, base_w in signals:
        weighted_signals.append((label, val, vote, base_w))
    signals = weighted_signals

    # ── Scoring ──────────────────────────────────────────────────────────────
    total_w  = sum(s[3] for s in signals) or 1
    bull_w   = sum(s[3] for s in signals if s[2] > 0)
    bear_w   = sum(s[3] for s in signals if s[2] < 0)
    active_w = bull_w + bear_w
    active_n = sum(1 for s in signals if s[2] != 0)
    net      = bull_w - bear_w

    agreement     = abs(net) / active_w if active_w else 0.0
    participation = active_w / total_w
    strength      = agreement * _math.sqrt(participation)

    if active_n >= _MIN_ACTIVE_SIGNALS and strength >= _MIN_STRENGTH:
        direction = "CALL" if net > 0 else "PUT"
    else:
        direction = "NO TRADE"

    # Weight-based conflict detection
    minority_share = min(bull_w, bear_w) / active_w if active_w else 0.0
    mixed = active_n >= _MIN_ACTIVE_SIGNALS and minority_share >= _MIXED_MINORITY_SHARE
    conflict_note = " ⚠️ MIXED SIGNALS" if mixed else ""

    # Catalyst adjustment — bounded so it nudges but can't manufacture a GO
    cat = cat_score if isinstance(cat_score, (int, float)) else _CAT_BASELINE
    cat_adj = max(-1.0, min((cat - _CAT_BASELINE) * _CAT_SLOPE, 1.5))

    conviction = max(0.0, min(10.0 * strength + cat_adj, 10.0))
    if mixed:
        conviction = min(conviction, _MIXED_CONVICTION_CAP)
    conviction = round(conviction, 1)

    # ── Structural gate + veto logic (Fable-recommended) ────────────────────
    _pm_vote    = next((s[2] for s in signals if s[0] == "Pre-Market Move"), 0)
    _vix_vote   = next((s[2] for s in signals if s[0] == "VIX Direction"), 0)
    _sec_vote   = next((s[2] for s in signals if s[0] == "Sector Pulse"), 0)
    _dxy_vote   = next((s[2] for s in signals if s[0] == "DXY"), 0)
    _fut_vote   = next((s[2] for s in signals if s[0] == "S&P Futures"), 0)
    _yield_vote = next((s[2] for s in signals if s[0] == "10Y Yield"), 0)

    # Gate: PM must be active AND (VIX Direction agrees OR both DXY+Futures agree)
    gate_pass = (_pm_vote != 0 and
                 (_pm_vote == _vix_vote or
                  (_pm_vote == _dxy_vote != 0 and _pm_vote == _fut_vote != 0)))

    # Super filter: PM + VIX Direction + Sector Pulse all agree (~63-68% live)
    super_filter = (_pm_vote != 0 and _pm_vote == _vix_vote == _sec_vote)

    # Yield veto: PM and 10Y pointing same direction = weak setup (47% backtest)
    yield_veto = (_yield_vote != 0 and _yield_vote == _pm_vote)

    # VIX term structure: backwardation blocks calls (IV crush risk)
    vix_ts = tech.get("vix_ts")
    vix_ts_block = (vix_ts is not None and vix_ts > 1.0 and direction == "CALL")

    # Earnings veto: skip ±1 day of earnings
    earnings_veto = tech.get("earnings_veto", False)

    # Tape contradiction: direction opposes BOTH VWAP and ORB (Fable hard veto)
    _vwap_vote = next((s[2] for s in signals if s[0] == "VWAP Position"), 0)
    _orb_vote  = next((s[2] for s in signals if s[0] == "ORB Direction"), 0)
    _dir_vote  = 1 if direction == "CALL" else (-1 if direction == "PUT" else 0)
    tape_contradiction = (_dir_vote != 0 and _vwap_vote != 0 and _orb_vote != 0 and
                          _dir_vote != _vwap_vote and _dir_vote != _orb_vote)

    # Blown-out range: OR wider than ATR — already exhausted (Fable hard veto)
    _or_atr_val    = tech.get("or_atr")
    blown_out_range = (_or_atr_val is not None and _or_atr_val > 1.0)

    # Dead tape: RVOL below 0.5 — no volume, avoid theta bleed
    dead_tape = (rvol is not None and rvol < 0.5)

    # ── GO / NO TRADE ────────────────────────────────────────────────────────
    go_threshold = _GO_THRESHOLD_PUT if direction == "PUT" else _GO_THRESHOLD_CALL
    is_go = (direction != "NO TRADE" and vix_ok
             and go_threshold <= conviction <= _GO_CONVICTION_MAX
             and gate_pass and not yield_veto
             and not vix_ts_block and not earnings_veto
             and not tape_contradiction and not blown_out_range
             and not dead_tape and not fomc_day)

    star = " ⭐ SUPER FILTER" if (is_go and super_filter) else ""
    go_nogo = f"GO{star}" if is_go else "NO TRADE"
    if mixed:
        go_nogo += " ⚠️ MIXED SIGNALS"
    if not vix_ok:
        go_nogo += f" ⚠️ VIX {regime} REGIME BLOCK"
    if not gate_pass and direction != "NO TRADE":
        go_nogo += " ⚠️ GATE FAIL"
    if yield_veto:
        go_nogo += " ⚠️ YIELD VETO"
    if vix_ts_block:
        go_nogo += " ⚠️ VIX BACKWARDATION"
    if earnings_veto:
        go_nogo += " ⚠️ EARNINGS WINDOW"
    if tape_contradiction:
        go_nogo += " ⚠️ TAPE CONTRADICTION"
    if blown_out_range:
        go_nogo += " ⚠️ BLOWN-OUT RANGE"
    if dead_tape:
        go_nogo += " ⚠️ DEAD TAPE"
    if fomc_day:
        go_nogo += " ⚠️ FOMC DAY"

    return direction, conviction, go_nogo, signals

# ── TERMINAL OUTPUT ───────────────────────────────────────────────────────
def print_report(tech, cross, sectors, catalysts, sentiment, sent_net,
                 signals, direction, conviction, go_nogo, chain, exp_date,
                 analyst=None):
    W = 62
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * W)
    print(f"  0DTE SPY SCANNER  |  {now}")
    print("=" * W)

    glyph = "✅  GO" if go_nogo == "GO" else "❌  NO TRADE"
    print(f"\n  {glyph}  —  {direction}  —  Conviction: {conviction}/10\n")

    print(f"── CATALYSTS {'─' * (W - 13)}")
    if catalysts:
        for c in catalysts:
            print(f"  • {c}")
    else:
        print("  None identified today")

    print(f"\n── MARKET TECHNICALS {'─' * (W - 21)}")
    if "spy_price" in tech:
        print(f"  SPY          ${tech['spy_price']:.2f}   Pre-mkt: {tech.get('spy_premkt', 0):+.2f}%")
    if "futures_price" in tech:
        print(f"  ES Futures   {tech['futures_price']:.0f}    Chg: {tech.get('futures_pct', 0):+.2f}%")
    if "vix" in tech:
        vix    = tech["vix"]
        note   = "✅ sweet spot" if 15 <= vix <= 28 else "⚠️  outside range"
        print(f"  VIX          {vix:.2f}  ({tech.get('vix_chg', 0):+.2f})  {note}")
    if "spy_rsi" in tech:
        print(f"  SPY RSI      {tech['spy_rsi']}")
    if "spy_pctb" in tech:
        print(f"  BB %%B        {tech['spy_pctb']}  (0=lower · 0.5=mid · 1=upper)")

    print(f"\n── CROSS-ASSET {'─' * (W - 15)}")
    for name, data in cross.items():
        arrow = "▲" if data["pct"] >= 0 else "▼"
        print(f"  {name:<14} {data['price']:>10.2f}   {arrow} {data['pct']:+.2f}%")

    print(f"\n── SECTOR PULSE {'─' * (W - 16)}")
    for name, pct in sectors.items():
        arrow = "▲" if pct >= 0 else "▼"
        print(f"  {name:<14} {arrow} {pct:+.2f}%")

    print(f"\n── SIGNAL BREAKDOWN {'─' * (W - 20)}")
    for label, val, vote, _ in signals:
        ind = "▲ CALL" if vote > 0 else ("▼ PUT " if vote < 0 else "— neut")
        print(f"  {label:<22} {val:<14} {ind}")

    print(f"\n── NEWS SENTIMENT: {sentiment} (net {sent_net:+d}) {'─' * (W - 32)}")

    if analyst:
        ups, dns, net, alabel, details = analyst
        print(f"\n── ANALYST RATINGS (last 7 days): {alabel} ({ups} up / {dns} dn) {'─' * max(0,W-52)}")
        for d in details:
            print(f"  {d}")

    print(f"\n── SPY 0DTE CHAIN  {exp_date or 'N/A'}  ({direction}) {'─' * max(0, W-37)}")
    if chain:
        print(f"  {'Strike':<8} {'Last':<8} {'Bid':<7} {'Ask':<7} {'Vol':<7} {'OI':<7} {'IV':<8} Delta")
        for row in chain:
            delta_s = f"{row.get('delta', 0):.3f}" if row.get('delta') is not None else "—"
            iv_s    = f"{row.get('impliedVolatility', 0)*100:.1f}%" if row.get('impliedVolatility') else "—"
            vol     = int(row.get('volume', 0) or 0)
            oi      = int(row.get('openInterest', 0) or 0)
            print(f"  ${row['strike']:<7.1f} ${row.get('lastPrice', 0):<6.2f} "
                  f"${row.get('bid', 0):<5.2f} ${row.get('ask', 0):<5.2f} "
                  f"{vol:<7} {oi:<7} {iv_s:<8} {delta_s}")
        best = chain[0]
        mid  = round((best.get("bid", 0) + best.get("ask", 0)) / 2, 2)
        cost = round(mid * 100, 2)
        print(f"\n  ★  RECOMMENDED: ${best['strike']:.0f} {direction} @ limit ${mid:.2f} (~${cost:.0f}/contract)")
    else:
        print("  Chain unavailable — market may not be open yet")

    print("\n" + "=" * W)
    print("  For informational purposes only — not financial advice")
    print("=" * W + "\n")

# ── HTML EMAIL ────────────────────────────────────────────────────────────
def _sec(title):
    NAV = "#1F4E79"
    return (f'<div style="font-size:13px;font-weight:bold;color:{NAV};'
            f'text-transform:uppercase;letter-spacing:1px;'
            f'border-bottom:2px solid {NAV};padding-bottom:6px;'
            f'margin-bottom:12px;">{title}</div>')

def format_html(tech, cross, sectors, catalysts, sentiment, sent_net,
                signals, direction, conviction, go_nogo, chain, exp_date,
                analyst=None):
    POS, NEG, NAV = "#276221", "#9C0006", "#1F4E79"
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    run_time  = datetime.now().strftime("%I:%M %p CT")

    go_bg    = "#1a7a1a" if go_nogo == "GO" else "#7a1a1a"
    go_label = (f"GO — {direction}" if go_nogo == "GO" else "NO TRADE — MIXED SIGNALS")

    # Catalysts
    cat_html = "".join(f'<li style="margin-bottom:6px;">{c}</li>' for c in catalysts) \
               or '<li style="margin-bottom:6px;color:#888;">None identified today</li>'

    # Signals table
    sig_rows = ""
    for i, (label, val, vote, _) in enumerate(signals):
        bg    = ' style="background:#f9f9f9;"' if i % 2 else ""
        col   = POS if vote > 0 else (NEG if vote < 0 else "#888")
        arrow = "▲ CALL" if vote > 0 else ("▼ PUT" if vote < 0 else "— neutral")
        sig_rows += (f'<tr{bg}>'
                     f'<td style="padding:6px 10px;">{label}</td>'
                     f'<td style="padding:6px 10px;">{val}</td>'
                     f'<td style="padding:6px 10px;color:{col};font-weight:bold;">{arrow}</td>'
                     f'</tr>')

    # Cross-asset table
    cross_rows = ""
    for i, (name, data) in enumerate(cross.items()):
        bg    = ' style="background:#f9f9f9;"' if i % 2 else ""
        col   = POS if data["pct"] >= 0 else NEG
        arrow = "▲" if data["pct"] >= 0 else "▼"
        cross_rows += (f'<tr{bg}>'
                       f'<td style="padding:6px 10px;">{name}</td>'
                       f'<td style="padding:6px 10px;text-align:right;">{data["price"]:.2f}</td>'
                       f'<td style="padding:6px 10px;text-align:right;color:{col};">'
                       f'{arrow} {data["pct"]:+.2f}%</td></tr>')

    # Sector table
    sector_rows = ""
    for i, (name, pct) in enumerate(sectors.items()):
        bg    = ' style="background:#f9f9f9;"' if i % 2 else ""
        col   = POS if pct >= 0 else NEG
        arrow = "▲" if pct >= 0 else "▼"
        sector_rows += (f'<tr{bg}>'
                        f'<td style="padding:6px 10px;">{name}</td>'
                        f'<td style="padding:6px 10px;text-align:right;color:{col};">'
                        f'{arrow} {pct:+.2f}%</td></tr>')

    # Technicals table
    vix = tech.get("vix", None)
    if vix:
        vix_note = ("✅ Sweet spot" if 15 <= vix <= 28
                    else ("⚠️ Too high — options expensive" if vix > 28
                          else "⚠️ Too low — limited movement"))
        vix_col  = POS if 15 <= vix <= 28 else NEG
    else:
        vix_note, vix_col = "—", "#888"

    tech_rows = (
        f'<tr><td style="padding:6px 10px;">SPY Price</td>'
        f'<td style="padding:6px 10px;text-align:right;">${tech.get("spy_price","—")}</td>'
        f'<td style="padding:6px 10px;text-align:right;color:{POS if tech.get("spy_premkt",0)>=0 else NEG};">'
        f'{tech.get("spy_premkt",0):+.2f}% pre-mkt</td></tr>'
        f'<tr style="background:#f9f9f9;"><td style="padding:6px 10px;">ES Futures</td>'
        f'<td style="padding:6px 10px;text-align:right;">{tech.get("futures_price","—")}</td>'
        f'<td style="padding:6px 10px;text-align:right;color:{POS if tech.get("futures_pct",0)>=0 else NEG};">'
        f'{tech.get("futures_pct",0):+.2f}%</td></tr>'
        f'<tr><td style="padding:6px 10px;">VIX</td>'
        f'<td style="padding:6px 10px;text-align:right;">{vix or "—"}</td>'
        f'<td style="padding:6px 10px;text-align:right;color:{vix_col};">{vix_note}</td></tr>'
        f'<tr style="background:#f9f9f9;"><td style="padding:6px 10px;">SPY RSI (14)</td>'
        f'<td style="padding:6px 10px;text-align:right;">{tech.get("spy_rsi","—")}</td>'
        f'<td style="padding:6px 10px;text-align:right;"></td></tr>'
        f'<tr><td style="padding:6px 10px;">BB %B</td>'
        f'<td style="padding:6px 10px;text-align:right;">{tech.get("spy_pctb","—")}</td>'
        f'<td style="padding:6px 10px;text-align:right;font-size:11px;color:#888;">'
        f'0=lower · 0.5=mid · 1=upper</td></tr>'
    )

    # Options chain
    chain_rows = ""
    best_rec   = ""
    if chain:
        for i, row in enumerate(chain):
            bg      = ' style="background:#f9f9f9;"' if i % 2 else ""
            delta_s = f"{row.get('delta', 0):.3f}" if row.get('delta') is not None else "—"
            iv_s    = f"{row.get('impliedVolatility', 0)*100:.1f}%" if row.get('impliedVolatility') else "—"
            bold    = "font-weight:bold;color:" + NAV + ";" if i == 0 else ""
            chain_rows += (f'<tr{bg}>'
                           f'<td style="padding:6px 8px;{bold}">${row["strike"]:.1f}</td>'
                           f'<td style="padding:6px 8px;text-align:right;">${row.get("lastPrice",0):.2f}</td>'
                           f'<td style="padding:6px 8px;text-align:right;">${row.get("bid",0):.2f}</td>'
                           f'<td style="padding:6px 8px;text-align:right;">${row.get("ask",0):.2f}</td>'
                           f'<td style="padding:6px 8px;text-align:right;">{int(row.get("volume",0) or 0)}</td>'
                           f'<td style="padding:6px 8px;text-align:right;">{int(row.get("openInterest",0) or 0)}</td>'
                           f'<td style="padding:6px 8px;text-align:right;">{iv_s}</td>'
                           f'<td style="padding:6px 8px;text-align:right;">{delta_s}</td></tr>')
        best = chain[0]
        mid  = round((best.get("bid", 0) + best.get("ask", 0)) / 2, 2)
        cost = round(mid * 100, 2)
        best_rec = (f'<div style="margin-top:12px;padding:12px 14px;background:#DCE6F1;'
                    f'border-radius:6px;font-size:13px;">'
                    f'★ <strong>Recommended Entry:</strong> '
                    f'${best["strike"]:.0f} {direction} @ limit '
                    f'<strong>${mid:.2f}</strong> (~${cost:.0f} per contract)</div>')
        stop_price  = round(mid * 0.50, 2)
        stop_cost   = round(stop_price * 100, 2)
        target_50   = round(mid * 1.50, 2)
        target_100  = round(mid * 2.00, 2)
        gain_50     = round(mid * 0.50 * 100, 2)
        gain_100    = round(mid * 1.00 * 100, 2)
        quick_entry_html = f"""
  <div style="margin:0;padding:16px 28px 20px;background:#fffbe6;border-top:3px solid #e6a800;">
    <div style="font-size:13px;font-weight:bold;color:#7a5500;text-transform:uppercase;
      letter-spacing:1px;border-bottom:2px solid #e6a800;padding-bottom:6px;margin-bottom:14px;">
      &#x1F4CB; Fidelity ATP Quick Entry
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#fff3cc;">
        <td style="padding:7px 10px;font-weight:bold;color:#555;width:40%;">Field</td>
        <td style="padding:7px 10px;font-weight:bold;color:#555;">Value</td>
      </tr>
      <tr><td style="padding:6px 10px;color:#555;">Symbol</td>
          <td style="padding:6px 10px;font-weight:bold;">SPY</td></tr>
      <tr style="background:#fff8e6;"><td style="padding:6px 10px;color:#555;">Action</td>
          <td style="padding:6px 10px;font-weight:bold;">BUY TO OPEN</td></tr>
      <tr><td style="padding:6px 10px;color:#555;">Option Type</td>
          <td style="padding:6px 10px;font-weight:bold;">{direction}</td></tr>
      <tr style="background:#fff8e6;"><td style="padding:6px 10px;color:#555;">Strike</td>
          <td style="padding:6px 10px;font-weight:bold;">${best["strike"]:.0f}</td></tr>
      <tr><td style="padding:6px 10px;color:#555;">Expiration</td>
          <td style="padding:6px 10px;font-weight:bold;">{exp_date or "Today"} &nbsp;<span style="color:#888;font-weight:normal;">(0DTE)</span></td></tr>
      <tr style="background:#fff8e6;"><td style="padding:6px 10px;color:#555;">Order Type</td>
          <td style="padding:6px 10px;font-weight:bold;">Limit</td></tr>
      <tr><td style="padding:6px 10px;color:#555;">Limit Price</td>
          <td style="padding:6px 10px;font-weight:bold;">${mid:.2f} &nbsp;<span style="color:#888;font-weight:normal;">(mid of ${best.get("bid",0):.2f} / ${best.get("ask",0):.2f})</span></td></tr>
      <tr style="background:#fff8e6;"><td style="padding:6px 10px;color:#555;">Cost / Contract</td>
          <td style="padding:6px 10px;font-weight:bold;">~${cost:.0f}</td></tr>
      <tr style="background:#fff8e6;"><td style="padding:6px 10px;color:#555;">Time in Force</td>
          <td style="padding:6px 10px;font-weight:bold;">Day</td></tr>
    </table>
    <div style="margin-top:14px;border-top:1px dashed #e6a800;padding-top:12px;">
      <div style="font-size:11px;font-weight:bold;color:#7a5500;text-transform:uppercase;
        letter-spacing:1px;margin-bottom:8px;">&#x1F504; OCO Bracket (place immediately after fill)</div>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <tr style="background:#fff3cc;">
          <td style="padding:5px 10px;font-weight:bold;color:#555;width:40%;">Order</td>
          <td style="padding:5px 10px;font-weight:bold;color:#555;">Price</td>
          <td style="padding:5px 10px;font-weight:bold;color:#555;">P&amp;L / Contract</td>
        </tr>
        <tr><td style="padding:6px 10px;color:#276221;font-weight:bold;">&#x1F3AF; Sell Limit (+100%)</td>
            <td style="padding:6px 10px;font-weight:bold;">${target_100:.2f}</td>
            <td style="padding:6px 10px;color:#276221;">+${gain_100:.0f}</td></tr>
        <tr style="background:#fff8e6;">
            <td style="padding:6px 10px;color:#276221;">&#x1F3AF; Sell Limit (+50%)</td>
            <td style="padding:6px 10px;font-weight:bold;">${target_50:.2f}</td>
            <td style="padding:6px 10px;color:#276221;">+${gain_50:.0f}</td></tr>
        <tr><td style="padding:6px 10px;color:#9C0006;font-weight:bold;">&#x1F6D1; Stop Market (-50%)</td>
            <td style="padding:6px 10px;font-weight:bold;">${stop_price:.2f} trigger</td>
            <td style="padding:6px 10px;color:#9C0006;">-${stop_cost:.0f}</td></tr>
      </table>
      <div style="margin-top:8px;font-size:11px;color:#888;font-style:italic;">
        Use Stop Market (not Stop Limit) for the loss side — 0DTE options can gap through a limit without filling.
      </div>
    </div>
  </div>"""
    else:
        quick_entry_html = ""
        chain_rows = ('<tr><td colspan="8" style="padding:8px 10px;color:#888;">'
                      'Chain unavailable — market may not be open yet</td></tr>')

    # Pre-compute analyst block (lambda-in-f-string not valid in Python < 3.12)
    if analyst:
        analyst_items = "".join(f'<li>{d}</li>' for d in analyst[4]) or "<li style='color:#888;'>No recent activity</li>"
        analyst_html = f"""
  <div style="padding:16px 28px 8px;">
    {_sec(f"Analyst Ratings — Last 7 Days ({analyst[3]})")}
    <div style="font-size:13px;margin-bottom:10px;color:#333;">
      <span style="color:{POS};font-weight:bold;">{analyst[0]} Upgrades</span>
      &nbsp;|&nbsp;
      <span style="color:{NEG};font-weight:bold;">{analyst[1]} Downgrades</span>
      &nbsp;|&nbsp; Net: <strong>{analyst[2]:+d}</strong>
    </div>
    <ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.8;color:#333;">
      {analyst_items}
    </ul>
  </div>"""
    else:
        analyst_html = ""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:Arial,sans-serif;">
<div style="max-width:640px;margin:24px auto;background:#fff;border-radius:8px;
  overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

  <div style="background:{NAV};padding:20px 28px;">
    <div style="color:#fff;font-size:18px;font-weight:bold;">0DTE SCANNER</div>
    <div style="color:#DCE6F1;font-size:13px;margin-top:4px;">{today_str}</div>
    <div style="color:#9ab5cc;font-size:11px;margin-top:2px;">Generated {run_time}</div>
  </div>

  <div style="background:{go_bg};padding:16px 28px;text-align:center;">
    <div style="color:#fff;font-size:17px;font-weight:bold;">{go_label}</div>
    <div style="color:rgba(255,255,255,0.75);font-size:13px;margin-top:4px;">
      Conviction: {conviction}/10</div>
  </div>

  {quick_entry_html}

  <div style="padding:20px 28px 8px;">
    {_sec("Today's Catalysts")}
    <ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.9;color:#333;">
      {cat_html}
    </ul>
  </div>

  <div style="padding:16px 28px 8px;">
    {_sec("Market Technicals")}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#DCE6F1;">
        <td style="padding:7px 10px;font-weight:bold;">Indicator</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Value</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Note</td>
      </tr>
      {tech_rows}
    </table>
  </div>

  <div style="padding:16px 28px 8px;">
    {_sec("Signal Breakdown")}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#DCE6F1;">
        <td style="padding:7px 10px;font-weight:bold;">Signal</td>
        <td style="padding:7px 10px;font-weight:bold;">Value</td>
        <td style="padding:7px 10px;font-weight:bold;">Vote</td>
      </tr>
      {sig_rows}
    </table>
  </div>

  <div style="padding:16px 28px 8px;">
    {_sec("Cross-Asset")}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#DCE6F1;">
        <td style="padding:7px 10px;font-weight:bold;">Asset</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Price</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Change</td>
      </tr>
      {cross_rows}
    </table>
  </div>

  {analyst_html}

  <div style="padding:16px 28px 8px;">
    {_sec("Sector Pulse")}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#DCE6F1;">
        <td style="padding:7px 10px;font-weight:bold;">Sector</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Move</td>
      </tr>
      {sector_rows}
    </table>
  </div>

  <div style="padding:16px 28px 8px;">
    {_sec(f"SPY 0DTE Chain — {exp_date or 'N/A'} ({direction})")}
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <tr style="background:#DCE6F1;">
        <td style="padding:6px 8px;font-weight:bold;">Strike</td>
        <td style="padding:6px 8px;font-weight:bold;text-align:right;">Last</td>
        <td style="padding:6px 8px;font-weight:bold;text-align:right;">Bid</td>
        <td style="padding:6px 8px;font-weight:bold;text-align:right;">Ask</td>
        <td style="padding:6px 8px;font-weight:bold;text-align:right;">Vol</td>
        <td style="padding:6px 8px;font-weight:bold;text-align:right;">OI</td>
        <td style="padding:6px 8px;font-weight:bold;text-align:right;">IV</td>
        <td style="padding:6px 8px;font-weight:bold;text-align:right;">Delta</td>
      </tr>
      {chain_rows}
    </table>
    {best_rec}
  </div>

  <div style="background:#f4f4f4;padding:12px 28px;font-size:11px;color:#888;
    text-align:center;border-top:1px solid #e0e0e0;margin-top:16px;">
    Generated by Claude &nbsp;|&nbsp; For informational purposes only — not financial advice
  </div>
</div></body></html>"""

# ── MULTI-TICKER EMAIL ───────────────────────────────────────────────────
def format_multi_html(results, market_tech, cross, sectors):
    NAV, POS, NEG = "#1F4E79", "#276221", "#9C0006"
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    run_time  = datetime.now().strftime("%I:%M %p CT")

    go_results    = [r for r in results if r.get("has_0dte") and r["go_nogo"] == "GO"]
    no_results    = [r for r in results if r.get("has_0dte") and r["go_nogo"] != "GO"]
    no_expiry     = [r for r in results if not r.get("has_0dte")]

    # Sort GO by conviction desc
    go_results.sort(key=lambda r: r["conviction"], reverse=True)

    # ── Market overview strip ─────────────────────────────────────────────
    vix     = market_tech.get("vix", "—")
    vix_chg = market_tech.get("vix_chg", 0)
    fut_pct = market_tech.get("futures_pct", 0)
    vix_col = POS if isinstance(vix, float) and 15 <= vix <= 28 else NEG
    fut_col = POS if fut_pct >= 0 else NEG

    overview_cells = (
        f'<td style="padding:10px 16px;text-align:center;">'
        f'<div style="font-size:11px;color:#9ab5cc;text-transform:uppercase;">VIX</div>'
        f'<div style="font-size:15px;font-weight:bold;color:{vix_col};">{vix}</div>'
        f'<div style="font-size:11px;color:#9ab5cc;">{vix_chg:+.2f} chg</div></td>'
        f'<td style="padding:10px 16px;text-align:center;">'
        f'<div style="font-size:11px;color:#9ab5cc;text-transform:uppercase;">ES Futures</div>'
        f'<div style="font-size:15px;font-weight:bold;color:{fut_col};">{fut_pct:+.2f}%</div>'
        f'<div style="font-size:11px;color:#9ab5cc;">overnight</div></td>'
    )
    for name, data in list(cross.items())[:3]:
        col = POS if data["pct"] >= 0 else NEG
        arr = "&#x25B2;" if data["pct"] >= 0 else "&#x25BC;"
        overview_cells += (
            f'<td style="padding:10px 16px;text-align:center;">'
            f'<div style="font-size:11px;color:#9ab5cc;text-transform:uppercase;">{name}</div>'
            f'<div style="font-size:15px;font-weight:bold;color:{col};">{arr} {data["pct"]:+.2f}%</div>'
            f'<div style="font-size:11px;color:#9ab5cc;">{data["price"]:.2f}</div></td>'
        )

    # ── GO ticker cards ───────────────────────────────────────────────────
    go_cards = ""
    for r in go_results:
        ticker    = r["ticker"]
        direction = r["direction"]
        conviction= r["conviction"]
        chain     = r.get("chain", [])
        exp_date  = r.get("exp_date", "Today")
        tech      = r.get("tech", {})
        price     = tech.get("tk_price", tech.get("spy_price", 0))
        premkt    = tech.get("tk_premkt", tech.get("spy_premkt", 0))
        sentiment = r.get("sentiment", "NEUTRAL")
        signals   = r.get("signals", [])

        bg       = "#1a7a1a" if direction == "CALL" else "#7a1a1a"
        dir_icon = "&#x25B2;" if direction == "CALL" else "&#x25BC;"
        p_col    = POS if premkt >= 0 else NEG

        # Signal rows (compact, 2 per row)
        sig_pairs = []
        for i in range(0, len(signals), 2):
            left  = signals[i]
            right = signals[i+1] if i+1 < len(signals) else None
            l_col = POS if left[2] > 0 else (NEG if left[2] < 0 else "#888")
            l_arr = "&#x25B2;" if left[2] > 0 else ("&#x25BC;" if left[2] < 0 else "&mdash;")
            row = (f'<tr>'
                   f'<td style="padding:4px 8px;font-size:12px;color:#555;">{left[0]}</td>'
                   f'<td style="padding:4px 8px;font-size:12px;">{left[1]}</td>'
                   f'<td style="padding:4px 8px;font-size:12px;color:{l_col};font-weight:bold;">{l_arr}</td>')
            if right:
                r_col = POS if right[2] > 0 else (NEG if right[2] < 0 else "#888")
                r_arr = "&#x25B2;" if right[2] > 0 else ("&#x25BC;" if right[2] < 0 else "&mdash;")
                row += (f'<td style="padding:4px 8px;font-size:12px;color:#555;border-left:1px solid #eee;">{right[0]}</td>'
                        f'<td style="padding:4px 8px;font-size:12px;">{right[1]}</td>'
                        f'<td style="padding:4px 8px;font-size:12px;color:{r_col};font-weight:bold;">{r_arr}</td>')
            else:
                row += '<td colspan="3"></td>'
            row += '</tr>'
            sig_pairs.append(row)

        # ATP quick entry (compact)
        atp_block = ""
        if chain:
            best      = chain[0]
            mid       = round((best.get("bid", 0) + best.get("ask", 0)) / 2, 2)
            cost      = round(mid * 100, 2)
            stop_p    = round(mid * 0.50, 2)
            stop_loss = round(stop_p * 100, 2)
            tgt_50    = round(mid * 1.50, 2)
            tgt_100   = round(mid * 2.00, 2)
            gain_50   = round(mid * 0.50 * 100, 2)
            gain_100  = round(mid * 1.00 * 100, 2)
            atp_block = f"""
    <div style="background:#fffbe6;padding:10px 14px;border-top:2px solid #e6a800;">
      <div style="font-size:11px;font-weight:bold;color:#7a5500;text-transform:uppercase;
        letter-spacing:1px;margin-bottom:8px;">&#x1F4CB; ATP Entry &amp; OCO Bracket</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <tr>
          <td style="padding:3px 6px;color:#555;width:38%;">Strike / Type</td>
          <td style="padding:3px 6px;font-weight:bold;">${best["strike"]:.0f} {direction} &nbsp;
            <span style="color:#888;font-weight:normal;">{exp_date} (0DTE)</span></td>
        </tr>
        <tr style="background:#fff8e6;">
          <td style="padding:3px 6px;color:#555;">Buy Limit</td>
          <td style="padding:3px 6px;font-weight:bold;">${mid:.2f}
            <span style="color:#888;font-weight:normal;">(~${cost:.0f}/contract)</span></td>
        </tr>
        <tr><td colspan="2" style="padding:4px 6px;font-size:11px;color:#b35900;
          font-style:italic;">&#x1F504; After fill, place OCO:</td></tr>
        <tr style="background:#f0fff0;">
          <td style="padding:3px 6px;color:#276221;font-weight:bold;">&#x1F3AF; Sell Limit +100%</td>
          <td style="padding:3px 6px;font-weight:bold;color:#276221;">${tgt_100:.2f}
            <span style="color:#888;font-weight:normal;font-weight:normal;">(+${gain_100:.0f}/contract)</span></td>
        </tr>
        <tr style="background:#f8fff8;">
          <td style="padding:3px 6px;color:#276221;">&#x1F3AF; Sell Limit +50%</td>
          <td style="padding:3px 6px;font-weight:bold;color:#276221;">${tgt_50:.2f}
            <span style="color:#888;font-weight:normal;">(+${gain_50:.0f}/contract)</span></td>
        </tr>
        <tr style="background:#fff5f5;">
          <td style="padding:3px 6px;color:#9C0006;font-weight:bold;">&#x1F6D1; Stop Market -50%</td>
          <td style="padding:3px 6px;font-weight:bold;color:#9C0006;">${stop_p:.2f} trigger
            <span style="color:#888;font-weight:normal;">(-${stop_loss:.0f}/contract)</span></td>
        </tr>
      </table>
    </div>"""

        go_cards += f"""
  <div style="border:2px solid {bg};border-radius:6px;margin:0 28px 16px;overflow:hidden;">
    <div style="background:{bg};padding:12px 16px;">
      <span style="color:#fff;font-size:15px;font-weight:bold;">
        {dir_icon} {ticker} &mdash; GO {direction}</span>
      <span style="float:right;color:rgba(255,255,255,0.8);font-size:13px;">
        Conviction: {conviction}/10</span>
    </div>
    <div style="padding:8px 14px;background:#fafafa;font-size:12px;color:#555;">
      Price: <strong>${price:.2f}</strong> &nbsp;
      Pre-mkt: <strong style="color:{p_col};">{premkt:+.2f}%</strong> &nbsp;
      News: <strong>{sentiment}</strong>
    </div>
    {atp_block}
    <div style="padding:10px 14px;">
      <table style="width:100%;border-collapse:collapse;">{''.join(sig_pairs)}</table>
    </div>
  </div>"""

    # ── NO TRADE tickers ─────────────────────────────────────────────────
    no_rows = ""
    for r in no_results:
        ticker    = r["ticker"]
        conviction= r["conviction"]
        direction = r["direction"]
        tech      = r.get("tech", {})
        premkt    = tech.get("tk_premkt", tech.get("spy_premkt", 0))
        p_col     = POS if premkt >= 0 else NEG
        no_rows += (f'<tr><td style="padding:6px 10px;">{ticker}</td>'
                    f'<td style="padding:6px 10px;">{direction}</td>'
                    f'<td style="padding:6px 10px;">{conviction}/10</td>'
                    f'<td style="padding:6px 10px;color:{p_col};">{premkt:+.2f}%</td></tr>')

    no_expiry_text = ""
    if no_expiry:
        names = ", ".join(r["ticker"] for r in no_expiry)
        no_expiry_text = (f'<div style="padding:8px 28px;font-size:12px;color:#888;">'
                          f'No 0DTE options today: {names}</div>')

    go_count = len(go_results)
    scan_count = len([r for r in results if r.get("has_0dte")])
    summary_col = POS if go_count > 0 else "#888"

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:Arial,sans-serif;">
<div style="max-width:680px;margin:24px auto;background:#fff;border-radius:8px;
  overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

  <div style="background:{NAV};padding:20px 28px;">
    <div style="color:#fff;font-size:18px;font-weight:bold;">0DTE MULTI-TICKER SCANNER</div>
    <div style="color:#DCE6F1;font-size:13px;margin-top:4px;">{today_str}</div>
    <div style="color:#9ab5cc;font-size:11px;margin-top:2px;">Generated {run_time}
      &nbsp;&mdash;&nbsp; {scan_count} tickers with 0DTE options
      &nbsp;&mdash;&nbsp; <span style="color:{summary_col};font-weight:bold;">{go_count} GO signal{'s' if go_count != 1 else ''}</span>
    </div>
  </div>

  <div style="background:#2a3f5a;padding:0;">
    <table style="width:100%;border-collapse:collapse;">{overview_cells}</table>
  </div>

  <div style="padding:20px 28px 8px;">
    {_sec("GO Signals" if go_count > 0 else "No GO Signals Today")}
  </div>

  {'<div style="padding:8px 28px 4px;font-size:13px;color:#888;">No GO signals this scan. All signals below conviction threshold or mixed direction.</div>' if go_count == 0 else go_cards}

  {'<div style="padding:16px 28px 8px;"><div style="font-size:13px;font-weight:bold;color:' + NAV + ';text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid ' + NAV + ';padding-bottom:6px;margin-bottom:14px;">No Trade</div><table style="width:100%;border-collapse:collapse;font-size:13px;"><tr style="background:#DCE6F1;"><td style="padding:7px 10px;font-weight:bold;">Ticker</td><td style="padding:7px 10px;font-weight:bold;">Direction</td><td style="padding:7px 10px;font-weight:bold;">Conviction</td><td style="padding:7px 10px;font-weight:bold;">Pre-mkt</td></tr>' + no_rows + '</table></div>' if no_rows else ''}

  {no_expiry_text}

  <div style="background:#f4f4f4;padding:12px 28px;font-size:11px;color:#888;
    text-align:center;border-top:1px solid #e0e0e0;margin-top:16px;">
    Generated by Claude &nbsp;|&nbsp; For informational purposes only — not financial advice
  </div>
</div></body></html>"""

# ── JSON SNAPSHOT ─────────────────────────────────────────────────────────
def write_json_snapshot(results, market_tech, cross, sectors, catalysts):
    import json
    go_results = [r for r in results if r.get("has_0dte") and "GO" in r.get("go_nogo", "")]
    go_results.sort(key=lambda r: r["conviction"], reverse=True)

    go_signals = []
    for r in go_results:
        chain    = r.get("chain", [])
        best_rec = None
        if chain:
            best = chain[0]
            mid  = round((best.get("bid", 0) + best.get("ask", 0)) / 2, 2)
            cost = round(mid * 100, 2)
            best_rec = {
                "strike":            best["strike"],
                "direction":         r["direction"],
                "limit":             mid,
                "cost_per_contract": cost,
                "stop":              round(mid * 0.50, 2),
                "target_50":         round(mid * 1.50, 2),
                "target_100":        round(mid * 2.00, 2),
            }
        tech = r.get("tech", {})
        go_signals.append({
            "ticker":     r["ticker"],
            "direction":  r["direction"],
            "conviction": r["conviction"],
            "go_nogo":    r["go_nogo"],
            "price":      tech.get("tk_price", tech.get("spy_price", 0)),
            "premkt":     tech.get("tk_premkt", tech.get("spy_premkt", 0)),
            "exp_date":   r.get("exp_date"),
            "signals":    [{"label": s[0], "value": s[1], "vote": s[2]} for s in r.get("signals", [])],
            "chain":      [{
                "strike":        row["strike"],
                "lastPrice":     row.get("lastPrice", 0),
                "bid":           row.get("bid", 0),
                "ask":           row.get("ask", 0),
                "volume":        int(row.get("volume", 0) or 0),
                "openInterest":  int(row.get("openInterest", 0) or 0),
                "iv":            round(row.get("impliedVolatility", 0) * 100, 1) if row.get("impliedVolatility") else None,
            } for row in chain],
            "recommended": best_rec,
        })

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "go_count":     len(go_signals),
        "market": {
            "vix":           market_tech.get("vix"),
            "vix_chg":       market_tech.get("vix_chg", 0),
            "futures_pct":   market_tech.get("futures_pct", 0),
            "futures_price": market_tech.get("futures_price"),
            "spy_price":     market_tech.get("spy_price"),
            "spy_premkt":    market_tech.get("spy_premkt", 0),
            "spy_rsi":       market_tech.get("spy_rsi"),
        },
        "cross":       cross,
        "sectors":     sectors,
        "catalysts":   catalysts,
        "go_signals":  go_signals,
        "all_results": [
            {
                "ticker":     r["ticker"],
                "direction":  r.get("direction", "—"),
                "conviction": r.get("conviction", 0),
                "go_nogo":    r.get("go_nogo", "—"),
                "has_0dte":   r.get("has_0dte", False),
            }
            for r in results
        ],
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_scan.json")
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"  Wrote latest_scan.json ({len(go_signals)} GO signals)")


# ── SEND EMAIL ────────────────────────────────────────────────────────────
def send_email(subject, html_body):
    if not GMAIL_APP_PASSWORD:
        print("No GMAIL_APP_PASSWORD — skipping email.")
        return
    msg = MIMEMultipart("alternative")
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(EMAIL_FROM, GMAIL_APP_PASSWORD)
        s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    print(f"{datetime.now():%Y-%m-%d %H:%M}  Running 0DTE multi-ticker scanner...")

    # Fetch market-wide data once
    catalysts, cat_score        = get_catalysts()
    market_tech                 = get_market_technicals()
    market_tech["put_call_ratio"] = get_put_call_ratio()
    cross                       = get_cross_asset()
    sectors                     = get_sector_pulse()

    # Scan each ticker
    results = []
    for ticker in SCAN_TICKERS:
        print(f"  Scanning {ticker}...", end=" ", flush=True)
        r = scan_ticker(ticker, market_tech, cross, sectors, cat_score)
        results.append(r)
        if not r["has_0dte"]:
            print("no 0DTE today")
        else:
            print(f"{r['go_nogo']} {r['direction']} ({r['conviction']}/10)")

    go_results = [r for r in results if r.get("has_0dte") and r["go_nogo"] == "GO"]
    go_results.sort(key=lambda r: r["conviction"], reverse=True)

    print(f"\n  {len(go_results)} GO signal(s) found")
    for r in go_results:
        print(f"  ★  {r['ticker']:6s} {r['direction']:4s}  Conviction {r['conviction']}/10")

    write_json_snapshot(results, market_tech, cross, sectors, catalysts)

    if SEND_EMAIL:
        html = format_multi_html(results, market_tech, cross, sectors)
        go_summary = (", ".join(f"{r['ticker']} {r['direction']}" for r in go_results)
                      if go_results else "No GO signals")
        subject = (f"\U0001f3af 0DTE Scanner — {go_summary} — "
                   f"{datetime.now().strftime('%a %b %d %I:%M %p CT')}")
        send_email(subject, html)
        print(f"{datetime.now():%Y-%m-%d %H:%M}  Emailed to {EMAIL_TO}")

if __name__ == "__main__":
    main()
