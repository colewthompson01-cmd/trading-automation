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
SCAN_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "META",
                "AMZN", "MSFT", "GOOGL", "AMD", "MU", "NFLX", "COIN"]

SECTORS = {
    "Tech":       "XLK",
    "Financials": "XLF",
    "Energy":     "XLE",
    "Healthcare": "XLV",
    "Comms":      "XLC",
    "Industrials":"XLI",
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
        vix = yf.Ticker("^VIX").history(period="2d")
        result["vix"]     = round(float(vix["Close"].iloc[-1]), 2)
        result["vix_prev"]= round(float(vix["Close"].iloc[-2]), 2)
        result["vix_chg"] = round(result["vix"] - result["vix_prev"], 2)
    except Exception:
        pass

    return result

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
        "futures_pct": market_tech.get("futures_pct", 0),
        "vix":         market_tech.get("vix"),
        "vix_chg":     market_tech.get("vix_chg", 0),
    }
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="30d")
        h1m  = t.history(period="1d", interval="1m", prepost=True)
        h2d  = t.history(period="2d")
        prev    = float(h2d["Close"].iloc[-2])
        current = float(h1m["Close"].iloc[-1])
        result["tk_price"]  = round(current, 2)
        result["tk_premkt"] = round((current / prev - 1) * 100, 2)
        result["tk_rsi"]    = calc_rsi(hist["Close"])
        result["tk_pctb"]   = calc_bb_pctb(hist["Close"])
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
def scan_ticker(ticker, market_tech, cross, sectors, cat_score):
    """Run full signal analysis for one ticker. Returns result dict."""
    exp_date = check_0dte_expiry(ticker)
    if not exp_date:
        return {"ticker": ticker, "has_0dte": False}

    tech               = get_ticker_technicals(ticker, market_tech)
    sentiment, sent_net = get_ticker_news(ticker)
    direction, conviction, go_nogo, signals = build_recommendation(
        tech, cross, sectors, sentiment, cat_score
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
def build_recommendation(tech, cross, sectors, sentiment, cat_score, analyst_net=0):
    signals = []  # (label, display_value, vote [-1/0/1], weight)

    if "futures_pct" in tech:
        fp   = tech["futures_pct"]
        vote = 1 if fp > 0.1 else (-1 if fp < -0.1 else 0)
        signals.append(("S&P Futures", f"{fp:+.2f}%", vote, 2))

    # Support both SPY-specific keys (single-ticker mode) and generic tk_* keys
    premkt = tech.get("tk_premkt", tech.get("spy_premkt"))
    if premkt is not None:
        vote = 1 if premkt > 0.1 else (-1 if premkt < -0.1 else 0)
        signals.append(("Pre-Market Move", f"{premkt:+.2f}%", vote, 2))

    if "vix_chg" in tech:
        vc   = tech["vix_chg"]
        vote = 1 if vc < -0.5 else (-1 if vc > 0.5 else 0)
        signals.append(("VIX Direction", f"{vc:+.2f}", vote, 1))

    if "10Y Yield" in cross:
        yp   = cross["10Y Yield"]["pct"]
        vote = -1 if yp > 0.5 else (1 if yp < -0.5 else 0)
        signals.append(("10Y Yield", f"{yp:+.2f}%", vote, 1))

    if "DXY" in cross:
        dp   = cross["DXY"]["pct"]
        vote = -1 if dp > 0.3 else (1 if dp < -0.3 else 0)
        signals.append(("DXY", f"{dp:+.2f}%", vote, 1))

    sent_vote = 1 if sentiment == "BULLISH" else (-1 if sentiment == "BEARISH" else 0)
    signals.append(("News Sentiment", sentiment, sent_vote, 1))

    if sectors:
        pos = sum(1 for v in sectors.values() if v > 0)
        neg = sum(1 for v in sectors.values() if v < 0)
        sec_vote = 1 if pos > neg else (-1 if neg > pos else 0)
        signals.append(("Sector Pulse", f"{pos} up / {neg} dn", sec_vote, 1))

    rsi = tech.get("tk_rsi", tech.get("spy_rsi"))
    if rsi is not None:
        vote = 1 if rsi < 45 else (-1 if rsi > 60 else 0)
        signals.append(("RSI (14)", str(rsi), vote, 1))

    pctb = tech.get("tk_pctb", tech.get("spy_pctb"))
    if pctb is not None:
        vote = -1 if pctb > 0.8 else (1 if pctb < 0.2 else 0)
        signals.append(("BB %B", str(pctb), vote, 1))

    if analyst_net != 0:
        a_vote = 1 if analyst_net > 0 else -1
        signals.append(("Analyst Ratings", f"{analyst_net:+d} net (7d)", a_vote, 0.5))

    # Dynamic weights based on VIX regime
    vix = tech.get("vix", 20)
    if isinstance(vix, (int, float)):
        if vix < 15:        # low vol — momentum signals matter more
            w_tech, w_sent = 1.5, 0.75
        elif vix > 28:      # high vol — price action outweighs sentiment
            w_tech, w_sent = 2.0, 0.5
        else:               # normal — equal weighting
            w_tech, w_sent = 1.0, 1.0
    else:
        w_tech, w_sent = 1.0, 1.0

    # Apply regime weights: scale technicals (futures, premkt, RSI, BB) and sentiment
    weighted_signals = []
    for label, val, vote, base_w in signals:
        if label in ("S&P Futures", "Pre-Market Move", "RSI (14)", "BB %B"):
            weighted_signals.append((label, val, vote, base_w * w_tech))
        elif label == "News Sentiment":
            weighted_signals.append((label, val, vote, base_w * w_sent))
        else:
            weighted_signals.append((label, val, vote, base_w))
    signals = weighted_signals

    total_w  = sum(s[3] for s in signals) or 1
    weighted = sum(s[2] * s[3] for s in signals)
    ratio    = weighted / total_w

    # Conflict detection — cap conviction when bulls and bears are roughly split
    bull_votes = sum(1 for s in signals if s[2] > 0)
    bear_votes = sum(1 for s in signals if s[2] < 0)
    mixed = abs(bull_votes - bear_votes) <= 1 and (bull_votes + bear_votes) >= 4
    conflict_note = " ⚠️ MIXED SIGNALS" if mixed else ""

    if ratio > 0.2:
        direction = "CALL"
    elif ratio < -0.2:
        direction = "PUT"
    else:
        direction = "NO TRADE"

    # VIX sweet-spot check
    vix_ok  = 15 <= vix <= 28 if isinstance(vix, (int, float)) else True
    vix_pen = 0 if vix_ok else -1.5

    conviction = round(min(abs(ratio) * 5 + cat_score + vix_pen, 10), 1)
    conviction = max(conviction, 0)
    if mixed:
        conviction = min(conviction, 6)  # cap at 6 on mixed signals
    go_nogo = "GO" if conviction >= 5 and direction != "NO TRADE" else "NO TRADE"
    if conflict_note:
        go_nogo = go_nogo + conflict_note

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
    catalysts, cat_score   = get_catalysts()
    market_tech            = get_market_technicals()
    cross                  = get_cross_asset()
    sectors                = get_sector_pulse()

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
