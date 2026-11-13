#!/usr/bin/env python3
"""
morning_briefing.py — Daily market briefing (pre- and post-market)
  Pre-market:  cron at 7:00 AM CT weekdays  →  python3 morning_briefing.py
  Post-market: cron at 4:30 PM CT weekdays  →  python3 morning_briefing.py --post
"""

import os
import sys
import subprocess
import smtplib
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import quote

POST_MODE = "--post" in sys.argv

import yfinance as yf

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

try:
    import pandas_market_calendars as mcal
    HAS_CALENDAR = True
except ImportError:
    HAS_CALENDAR = False

# ── CONFIG ───────────────────────────────────────────────────────────────
EMAIL_TO           = "cole.w.thompson.01@gmail.com"
EMAIL_FROM         = "cole.w.thompson.01@gmail.com"
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
SCREENER_PATH      = os.path.expanduser("~/Downloads/bb_screener.py")

PORTFOLIO = [
    "VST", "CRDO", "CRWV", "SMCI", "MU", "AMAT", "NEM", "META",
    "SPY", "LUV", "FCNTX", "FXAIX", "SDCI", "AEP", "PINE",
    "AMZN", "UHS", "TXT", "THG", "GM", "CRSR", "HALO",
]

# Tickers to always flag regardless of price action (active thesis, open position, etc.)
# Edit this list when you have a live trade or setup in mind; leave empty for fully auto.
PINNED = []

MACRO_TICKERS = {
    "S&P 500":      ("^GSPC",    False),
    "Nasdaq":       ("^IXIC",    False),
    "Dow Jones":    ("^DJI",     False),
    "Russell 2000": ("^RUT",     False),
    "10Y Treasury": ("^TNX",     True),   # yield — show as X.XX%
    "Short Rate":   ("^IRX",     True),
    "DXY":          ("DX-Y.NYB", False),
    "WTI Crude":    ("CL=F",     False),
    "Gold":         ("GC=F",     False),
}

# ── DYNAMIC WATCHLIST ────────────────────────────────────────────────────
def build_watchlist(movers, earnings):
    """Top movers + earnings-today tickers + any manually pinned names."""
    watch = [t for t, _, _ in movers[:4]]
    for ticker, label, _ in earnings:
        if label == "TODAY" and ticker not in watch:
            watch.append(ticker)
    for t in PINNED:
        if t not in watch:
            watch.append(t)
    return watch

# ── MARKET CALENDAR ───────────────────────────────────────────────────────
def is_market_open_today():
    if not HAS_CALENDAR:
        return date.today().weekday() < 5
    nyse = mcal.get_calendar("NYSE")
    today_str = date.today().strftime("%Y-%m-%d")
    return not nyse.schedule(start_date=today_str, end_date=today_str).empty

# ── MACRO SNAPSHOT ────────────────────────────────────────────────────────
def get_macro_snapshot():
    rows = []
    for name, (sym, is_yield) in MACRO_TICKERS.items():
        try:
            hist = yf.Ticker(sym).history(period="2d")
            if len(hist) < 2:
                continue
            prev = hist["Close"].iloc[-2]
            last = hist["Close"].iloc[-1]
            pct  = (last / prev - 1) * 100
            rows.append({
                "name":     name,
                "level":    last,
                "pct":      round(pct, 2),
                "is_yield": is_yield,
            })
        except Exception:
            pass
    return rows

# ── PORTFOLIO MOVERS ──────────────────────────────────────────────────────
def get_movers(tickers):
    movers = []
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if len(hist) < 2:
                continue
            prev = hist["Close"].iloc[-2]
            last = hist["Close"].iloc[-1]
            pct  = (last / prev - 1) * 100
            if abs(pct) >= 0.5:
                movers.append((ticker, round(pct, 2), round(last, 2)))
        except Exception:
            pass
    movers.sort(key=lambda x: abs(x[1]), reverse=True)
    return movers[:10]

# ── NEWS helpers ──────────────────────────────────────────────────────────
def _rss_fetch(query, max_items=4):
    """Returns list of (title, url) tuples."""
    stories, seen = [], set()
    try:
        feed_url = (f"https://news.google.com/rss/search"
                    f"?q={quote(query)}&hl=en-US&gl=US&ceid=US:en")
        req = urllib.request.Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8, context=_SSL_CTX) as r:
            root = ET.fromstring(r.read())
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            link  = item.findtext("link", "").strip()
            if " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()
            if title and title not in seen:
                seen.add(title)
                stories.append((title, link))
            if len(stories) >= max_items:
                break
    except Exception:
        pass
    return stories

def get_market_drivers():
    return [title for title, _ in _rss_fetch("US stock market today", max_items=4)]

def _get_news_yfinance(tickers, priority=None):
    stories, seen = [], set()
    priority = set(priority or [])
    ordered  = [t for t in tickers if t in priority] + \
               [t for t in tickers if t not in priority]
    for ticker in ordered:
        try:
            for item in (yf.Ticker(ticker).news or [])[:2]:
                title = item.get("title", "").strip()
                url   = item.get("link", item.get("url", "")).strip()
                if title and title not in seen:
                    seen.add(title)
                    stories.append((ticker, ticker, title, ticker in priority, url))
        except Exception:
            pass
        if len(stories) >= 15:
            break
    return stories

def _get_news_rss(tickers, priority=None):
    stories, seen = [], set()
    priority = set(priority or [])
    for ticker in tickers:
        for title, url in _rss_fetch(f"{ticker} stock", max_items=2):
            if title not in seen:
                seen.add(title)
                stories.append((ticker, ticker, title, ticker in priority, url))
    return stories

def get_news(tickers, watchlist=None):
    yf_raw      = _get_news_yfinance(tickers, priority=watchlist)
    rss_tickers = (watchlist or []) + [t for t in tickers if t not in set(watchlist or [])]
    rss_raw     = _get_news_rss(rss_tickers[:10] if len(yf_raw) >= 5 else tickers, priority=watchlist)
    seen, combined = set(), []
    for item in rss_raw + yf_raw:
        key = item[2].lower()[:60]
        if key not in seen:
            seen.add(key)
            combined.append(item)
        if len(combined) >= 15:
            break
    return combined

# ── EARNINGS THIS WEEK ────────────────────────────────────────────────────
def get_earnings_this_week(tickers):
    upcoming = []
    today    = date.today()
    week_end = today + timedelta(days=7)
    for ticker in tickers:
        try:
            cal = yf.Ticker(ticker).calendar
            if cal is None:
                continue
            dates = cal.get("Earnings Date", []) if isinstance(cal, dict) else []
            if not isinstance(dates, list):
                dates = [dates]
            for ed in dates:
                ed_date = ed.date() if hasattr(ed, "date") else ed
                if isinstance(ed_date, date) and today <= ed_date <= week_end:
                    label   = "TODAY" if ed_date == today else ed_date.strftime("%a %b %d")
                    is_w    = ticker in set(WATCHLIST)
                    upcoming.append((ticker, label, is_w))
        except Exception:
            pass
    return upcoming

# ── BB SCREENER ───────────────────────────────────────────────────────────
def run_screener():
    try:
        result = subprocess.run(
            ["python3", SCREENER_PATH],
            capture_output=True, text=True, timeout=360
        )
        output = result.stdout
        return output[output.index("FINAL RESULTS"):] if "FINAL RESULTS" in output \
               else output[-2000:]
    except Exception as e:
        return f"Screener error: {e}"

# ── SECTION HEADER (Financial Brief style) ────────────────────────────────
def _sec(title):
    return (f'<div style="font-size:13px;font-weight:bold;color:#1F4E79;'
            f'text-transform:uppercase;letter-spacing:1px;'
            f'border-bottom:2px solid #1F4E79;padding-bottom:6px;'
            f'margin-bottom:12px;">{title}</div>')

# ── FORMAT HTML ───────────────────────────────────────────────────────────
def format_briefing_html(macro, movers, drivers, news, earnings, screener_output, watchlist=None, post=False):
    today_str    = datetime.now().strftime("%A, %B %d, %Y")
    run_time     = datetime.now().strftime("%I:%M %p CT")
    watch_set    = set(watchlist or [])
    header_title = "CLOSING BRIEF" if post else "MORNING BRIEFING"
    news_section = "Today's Headlines" if post else "Overnight News"
    drivers_section = "What Drove Today's Session" if post else "What Drove It"
    POS, NEG, NEU = "#276221", "#9C0006", "#333333"

    # ── Macro snapshot table ──
    macro_rows = ""
    for i, row in enumerate(macro):
        bg    = ' style="background:#f9f9f9;"' if i % 2 else ""
        up    = row["pct"] >= 0
        col   = POS if up else NEG
        arrow = "▲" if up else "▼"
        if row["is_yield"]:
            level_str = f"{row['level']:.2f}%"
            chg_str   = f"{row['pct']:+.1f}bps"
        else:
            level_str = f"{row['level']:,.2f}"
            chg_str   = f"{arrow} {row['pct']:+.1f}%"
        macro_rows += (
            f'<tr{bg}>'
            f'<td style="padding:6px 10px;">{row["name"]}</td>'
            f'<td style="padding:6px 10px;text-align:right;">{level_str}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{col};">{chg_str}</td>'
            f'</tr>'
        )

    # ── Portfolio movers table ──
    mover_rows = ""
    for i, (ticker, pct, price) in enumerate(movers):
        bg      = ' style="background:#f9f9f9;"' if i % 2 else ""
        up      = pct >= 0
        col     = POS if up else NEG
        arrow   = "▲" if up else "▼"
        is_w    = ticker in watch_set
        is_big  = abs(pct) > 4
        tkr_style = ("font-weight:bold;color:#1F4E79;" if is_w
                     else "font-weight:bold;color:#333333;")
        badge   = (' <span style="background:#C55A11;color:#fff;font-size:10px;'
                   'font-weight:bold;padding:1px 5px;border-radius:3px;">'
                   'BIG MOVE</span>' if is_big else "")
        mover_rows += (
            f'<tr{bg}>'
            f'<td style="padding:6px 10px;{tkr_style}">{ticker}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{col};">'
            f'{arrow} {pct:+.1f}%{badge}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:#555;">'
            f'${price:.2f}</td>'
            f'</tr>'
        )
    if not mover_rows:
        mover_rows = ('<tr><td colspan="3" style="padding:8px 10px;color:#888;">'
                      'No moves &gt;0.5%</td></tr>')

    # ── What Drove It ──
    drivers_html = ""
    if drivers:
        for d in drivers[:4]:
            drivers_html += (f'<li style="margin-bottom:6px;">{d}</li>')
    else:
        drivers_html = '<li style="margin-bottom:6px;">No major macro drivers found</li>'

    # ── Earnings ──
    earn_rows = ""
    if earnings:
        for ticker, label, is_w in earnings:
            col   = "#1F4E79" if is_w else NEU
            badge = (' <span style="color:#C55A11;font-weight:bold;">⚠️</span>'
                     if is_w else "")
            earn_rows += (f'<li style="margin-bottom:6px;color:{col};">'
                          f'<strong>{ticker}</strong> — {label}{badge}</li>')
    else:
        earn_rows = '<li style="margin-bottom:6px;">None for your holdings</li>'

    # ── BB Screener ──
    calls, puts, setup_lines = 0, 0, []
    for line in screener_output.strip().splitlines():
        if "Call setups" in line:
            try: calls = int("".join(c for c in line if c.isdigit()))
            except: pass
        elif "Put setups" in line:
            try: puts = int("".join(c for c in line if c.isdigit()))
            except: pass
        elif "|" in line and "Score" in line:
            setup_lines.append(line.strip())

    screener_rows = ""
    for i, s in enumerate(setup_lines):
        bg     = ' style="background:#f9f9f9;"' if i % 2 else ""
        parts  = [p.strip() for p in s.split("|")]
        tkr    = parts[0] if parts else "—"
        is_c   = i < calls
        col    = POS if is_c else NEG
        label  = "CALL ▲" if is_c else "PUT ▼"
        detail = "  ·  ".join(parts[1:]) if len(parts) > 1 else ""
        screener_rows += (
            f'<tr{bg}>'
            f'<td style="padding:6px 10px;font-weight:bold;color:{col};">{label}</td>'
            f'<td style="padding:6px 10px;font-weight:bold;">{tkr}</td>'
            f'<td style="padding:6px 10px;color:#555;font-size:12px;">{detail}</td>'
            f'</tr>'
        )
    if not screener_rows:
        screener_rows = ('<tr><td colspan="3" style="padding:8px 10px;color:#888;">'
                         'No setups today</td></tr>')
    screener_summary = (
        f'<span style="color:{POS};font-weight:bold;">{calls} Call'
        f'{"s" if calls != 1 else ""}</span>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<span style="color:{NEG};font-weight:bold;">{puts} Put'
        f'{"s" if puts != 1 else ""}</span>'
    )

    # ── News ──
    news_rows = ""
    if news:
        for i, (flag, ticker, headline, is_w, url) in enumerate(news[:12]):
            col     = "#1F4E79" if is_w else NEU
            tkr_tag = (f'<span style="font-weight:bold;color:{col};'
                       f'margin-right:6px;">[{ticker}]</span>')
            if url:
                hl_tag = (f'<a href="{url}" style="color:#1155CC;text-decoration:none;"'
                          f' target="_blank">{headline}</a>')
            else:
                hl_tag = f'<span style="color:{NEU};">{headline}</span>'
            news_rows += (
                f'<li style="margin-bottom:7px;">'
                f'{tkr_tag}{hl_tag}'
                f'</li>'
            )
    else:
        news_rows = '<li style="margin-bottom:6px;">No major headlines</li>'

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:Arial,sans-serif;">
<div style="max-width:600px;margin:24px auto;background:#ffffff;
  border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

  <!-- HEADER -->
  <div style="background:#1F4E79;padding:20px 28px;">
    <div style="color:#ffffff;font-size:18px;font-weight:bold;">
      {header_title}</div>
    <div style="color:#DCE6F1;font-size:13px;margin-top:4px;">{today_str}</div>
    <div style="color:#9ab5cc;font-size:11px;margin-top:2px;">Generated {run_time}</div>
  </div>

  <!-- MARKET SNAPSHOT -->
  <div style="padding:20px 28px 8px;">
    {_sec("Market Snapshot")}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#DCE6F1;">
        <td style="padding:7px 10px;font-weight:bold;">Index / Asset</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Level</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Change</td>
      </tr>
      {macro_rows}
    </table>
  </div>

  <!-- PORTFOLIO MOVERS -->
  <div style="padding:16px 28px 8px;">
    {_sec("Portfolio Movers")}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#DCE6F1;">
        <td style="padding:7px 10px;font-weight:bold;">Ticker</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Move</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Price</td>
      </tr>
      {mover_rows}
    </table>
  </div>

  <!-- WHAT DROVE IT -->
  <div style="padding:16px 28px 8px;">
    {_sec(drivers_section)}
    <ul style="margin:0;padding-left:18px;font-size:13px;
      line-height:1.7;color:#333333;">
      {drivers_html}
    </ul>
  </div>

  <!-- EARNINGS THIS WEEK -->
  <div style="padding:16px 28px 8px;">
    {_sec("Earnings This Week")}
    <ul style="margin:0;padding-left:18px;font-size:13px;
      line-height:1.7;color:#333333;">
      {earn_rows}
    </ul>
  </div>

  <!-- BB SCREENER -->
  <div style="padding:16px 28px 8px;">
    {_sec("BB Screener — Today's Setups")}
    <div style="font-size:13px;margin-bottom:10px;">{screener_summary}</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#DCE6F1;">
        <td style="padding:7px 10px;font-weight:bold;">Signal</td>
        <td style="padding:7px 10px;font-weight:bold;">Ticker</td>
        <td style="padding:7px 10px;font-weight:bold;">Details</td>
      </tr>
      {screener_rows}
    </table>
  </div>

  <!-- NEWS -->
  <div style="padding:16px 28px 8px;">
    {_sec(news_section)}
    <ul style="margin:0;padding-left:18px;font-size:13px;
      line-height:1.7;color:#333333;">
      {news_rows}
    </ul>
  </div>

  <!-- FOOTER -->
  <div style="background:#f4f4f4;padding:12px 28px;font-size:11px;
    color:#888888;text-align:center;border-top:1px solid #e0e0e0;
    margin-top:16px;">
    Generated by Claude &nbsp;|&nbsp; For informational purposes only
  </div>

</div></body></html>"""


# ── SEND EMAIL ────────────────────────────────────────────────────────────
def send_email(subject, body):
    if not GMAIL_APP_PASSWORD:
        print("No GMAIL_APP_PASSWORD set — printing briefing (not emailed):\n")
        print(body[:500], "...")
        return
    msg = MIMEMultipart("alternative")
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, GMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    if not is_market_open_today():
        print(f"{datetime.now():%Y-%m-%d %H:%M}  Market closed today — skipping.")
        return

    label = "closing brief" if POST_MODE else "morning briefing"
    print(f"{datetime.now():%Y-%m-%d %H:%M}  Generating {label}...")

    all_tickers = list(dict.fromkeys(PINNED + PORTFOLIO))

    macro     = get_macro_snapshot()
    movers    = get_movers(all_tickers)
    earnings  = get_earnings_this_week(all_tickers)
    watchlist = build_watchlist(movers, earnings)
    drivers   = get_market_drivers()
    news      = get_news(all_tickers, watchlist=watchlist)
    screener  = run_screener()

    body = format_briefing_html(
        macro, movers, drivers, news, earnings, screener, watchlist, post=POST_MODE
    )
    subject = (f"📈 Market Close — {datetime.now().strftime('%a %b %d')}"
               if POST_MODE else
               f"📊 Morning Briefing — {datetime.now().strftime('%a %b %d')}")

    send_email(subject, body)
    print(f"{datetime.now():%Y-%m-%d %H:%M}  {label.capitalize()} emailed to {EMAIL_TO}")

if __name__ == "__main__":
    main()
