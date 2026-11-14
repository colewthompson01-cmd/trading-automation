#!/usr/bin/env python3
"""
exit_scanner.py — 0DTE position exit monitor

Usage:
  python3 exit_scanner.py SPY 750 PUT 2.27          # 1 contract
  python3 exit_scanner.py SPY 750 PUT 2.27 5        # 5 contracts
  python3 exit_scanner.py NVDA 135 CALL 1.50 2
"""

import os, sys, smtplib
from datetime import datetime, date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import yfinance as yf

EMAIL_TO           = "cole.w.thompson.01@gmail.com"
EMAIL_FROM         = "cole.w.thompson.01@gmail.com"
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")

# ── FETCH ─────────────────────────────────────────────────────────────────
def get_option(ticker, strike, direction):
    try:
        t    = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None
        today = date.today().strftime("%Y-%m-%d")
        exp   = today if today in exps else exps[0]
        chain = t.option_chain(exp)
        df    = chain.calls if direction == "CALL" else chain.puts
        row   = df[df["strike"] == float(strike)]
        if row.empty:
            df["_d"] = abs(df["strike"] - float(strike))
            row = df.nsmallest(1, "_d")
        if row.empty:
            return None
        r   = row.iloc[0]
        bid = float(r.get("bid", 0) or 0)
        ask = float(r.get("ask", 0) or 0)
        mid = round((bid + ask) / 2, 2) if bid and ask else float(r.get("lastPrice", 0) or 0)
        return {
            "bid": bid, "ask": ask, "mid": mid,
            "last": float(r.get("lastPrice", 0) or 0),
            "iv":    r.get("impliedVolatility"),
            "delta": r.get("delta"),
            "theta": r.get("theta"),
            "exp":   exp,
            "strike": float(r["strike"]),
        }
    except Exception as e:
        return {"error": str(e)}

def get_underlying(ticker):
    try:
        hist  = yf.Ticker(ticker).history(period="1d", interval="1m")
        price = float(hist["Close"].iloc[-1])
        pct   = round((price / float(hist["Open"].iloc[0]) - 1) * 100, 2)
        return price, pct
    except Exception:
        return None, None

# ── HTML ──────────────────────────────────────────────────────────────────
def _sec(title, color="#1F4E79"):
    return (f'<div style="font-size:13px;font-weight:bold;color:{color};'
            f'text-transform:uppercase;letter-spacing:1px;'
            f'border-bottom:2px solid {color};padding-bottom:6px;'
            f'margin-bottom:14px;">{title}</div>')

def build_html(ticker, strike, direction, entry, contracts, opt, u_price, u_pct):
    NAV, POS, NEG = "#1F4E79", "#276221", "#9C0006"
    now      = datetime.now()
    time_str = now.strftime("%I:%M %p CT")

    if not opt or "error" in opt:
        err = opt.get("error", "Could not fetch option data") if opt else "No data returned"
        body = f'<div style="padding:20px 28px;color:{NEG};">Error: {err}</div>'
        status_bg, banner = "#7a1a1a", "ERROR — could not fetch option data"
    else:
        mid  = opt["mid"]
        bid  = opt["bid"]
        ask  = opt["ask"]

        pnl_per   = round((mid - entry) * 100, 2)
        pnl_total = round(pnl_per * contracts, 2)
        pnl_pct   = round((mid - entry) / entry * 100, 1) if entry else 0
        pnl_col   = POS if pnl_total >= 0 else NEG
        arrow     = "▲" if pnl_total >= 0 else "▼"

        t50  = round(entry * 1.50, 2)
        t100 = round(entry * 2.00, 2)
        stop = round(entry * 0.50, 2)

        if mid >= t100:
            status_bg = "#1a5c1a"
            banner    = f"&#x1F3AF; AT 100% TARGET — Consider taking profit"
        elif mid >= t50:
            status_bg = "#2a7a2a"
            banner    = f"&#x1F3AF; AT 50% TARGET — Consider partial exit"
        elif mid <= stop:
            status_bg = "#7a1a1a"
            banner    = f"&#x1F6D1; AT STOP — Consider closing position"
        else:
            status_bg = NAV
            banner    = f"&#x1F4CA; IN POSITION — {pnl_pct:+.1f}%"

        # Time warning (CST)
        hour = now.hour
        if hour >= 15:
            warn_msg = "&#x1F6A8; Market close in &lt;1 hour — close position or let expire"
            warn_col = NEG
        elif hour >= 14:
            warn_msg = "&#x26A0;&#xFE0F; 1 hour to close — theta accelerating fast"
            warn_col = "#b35900"
        elif hour >= 13:
            warn_msg = "&#x26A0;&#xFE0F; 2 hours to close — theta picking up"
            warn_col = "#b35900"
        else:
            warn_msg = "&#x2705; Plenty of time remaining"
            warn_col = POS

        iv_s    = f"{opt['iv']*100:.1f}%" if opt.get("iv") else "—"
        delta_s = f"{opt['delta']:.3f}"   if opt.get("delta") is not None else "—"
        theta_s = f"{opt['theta']:.3f}"   if opt.get("theta") is not None else "—"

        u_col  = POS if (u_pct or 0) >= 0 else NEG
        u_arr  = "&#x25B2;" if (u_pct or 0) >= 0 else "&#x25BC;"

        def hit(cond_hit, hit_label, miss_label, hit_col):
            return (f'<span style="color:{hit_col};font-weight:bold;">{hit_label}</span>'
                    if cond_hit else
                    f'<span style="color:#888;">{miss_label}</span>')

        body = f"""
  <div style="background:{status_bg};padding:16px 28px;text-align:center;">
    <div style="color:#fff;font-size:17px;font-weight:bold;">{banner}</div>
    <div style="color:rgba(255,255,255,0.75);font-size:13px;margin-top:4px;">
      {ticker} ${strike:.0f} {direction} &nbsp;|&nbsp; Entry: ${entry:.2f} &nbsp;|&nbsp; Now: ${mid:.2f}
    </div>
  </div>

  <div style="padding:20px 28px 8px;">
    {_sec("P&amp;L Summary")}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr>
        <td style="padding:8px 10px;color:#555;">Entry Price</td>
        <td style="padding:8px 10px;font-weight:bold;">${entry:.2f}
          <span style="color:#888;font-size:12px;">(~${entry*100:.0f}/contract)</span></td>
      </tr>
      <tr style="background:#f9f9f9;">
        <td style="padding:8px 10px;color:#555;">Current Mid</td>
        <td style="padding:8px 10px;font-weight:bold;">${mid:.2f}
          <span style="color:#888;font-size:12px;">(bid ${bid:.2f} / ask ${ask:.2f})</span></td>
      </tr>
      <tr>
        <td style="padding:8px 10px;color:#555;">P&amp;L per Contract</td>
        <td style="padding:8px 10px;font-weight:bold;color:{pnl_col};">
          {arrow} ${abs(pnl_per):.2f}
          <span style="font-size:12px;">({pnl_pct:+.1f}%)</span></td>
      </tr>
      <tr style="background:#f9f9f9;">
        <td style="padding:8px 10px;color:#555;">Total P&amp;L ({contracts} contract{'s' if contracts != 1 else ''})</td>
        <td style="padding:8px 10px;font-size:16px;font-weight:bold;color:{pnl_col};">
          {arrow} ${abs(pnl_total):.2f}</td>
      </tr>
      <tr>
        <td style="padding:8px 10px;color:#555;">Underlying ({ticker})</td>
        <td style="padding:8px 10px;font-weight:bold;">
          {'$'+f'{u_price:.2f}' if u_price else '—'}
          <span style="color:{u_col};font-size:12px;">
            {u_arr} {u_pct:+.2f}% today</span></td>
      </tr>
    </table>
  </div>

  <div style="padding:16px 28px 8px;">
    {_sec("Exit Targets")}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr style="background:#DCE6F1;">
        <td style="padding:7px 10px;font-weight:bold;">Level</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Option Price</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Gain/Contract</td>
        <td style="padding:7px 10px;font-weight:bold;text-align:right;">Status</td>
      </tr>
      <tr>
        <td style="padding:6px 10px;">&#x1F3AF; +100% Target</td>
        <td style="padding:6px 10px;text-align:right;">${t100:.2f}</td>
        <td style="padding:6px 10px;text-align:right;color:{POS};">+${entry*100:.0f}</td>
        <td style="padding:6px 10px;text-align:right;">
          {hit(mid >= t100, '&#x2705; HIT', 'Not yet', POS)}</td>
      </tr>
      <tr style="background:#f9f9f9;">
        <td style="padding:6px 10px;">&#x1F3AF; +50% Target</td>
        <td style="padding:6px 10px;text-align:right;">${t50:.2f}</td>
        <td style="padding:6px 10px;text-align:right;color:{POS};">+${entry*50:.0f}</td>
        <td style="padding:6px 10px;text-align:right;">
          {hit(mid >= t50, '&#x2705; HIT', 'Not yet', POS)}</td>
      </tr>
      <tr>
        <td style="padding:6px 10px;">&#x1F6D1; Stop (-50%)</td>
        <td style="padding:6px 10px;text-align:right;">${stop:.2f}</td>
        <td style="padding:6px 10px;text-align:right;color:{NEG};">-${entry*50:.0f}</td>
        <td style="padding:6px 10px;text-align:right;">
          {hit(mid <= stop, '&#x1F6D1; HIT', 'Safe', NEG)}</td>
      </tr>
    </table>
  </div>

  <div style="padding:16px 28px 8px;">
    {_sec("Greeks &amp; IV")}
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <tr>
        <td style="padding:6px 10px;color:#555;">IV</td>
        <td style="padding:6px 10px;font-weight:bold;">{iv_s}</td>
      </tr>
      <tr style="background:#f9f9f9;">
        <td style="padding:6px 10px;color:#555;">Delta</td>
        <td style="padding:6px 10px;font-weight:bold;">{delta_s}</td>
      </tr>
      <tr>
        <td style="padding:6px 10px;color:#555;">Theta (daily)</td>
        <td style="padding:6px 10px;font-weight:bold;color:{NEG};">{theta_s}</td>
      </tr>
    </table>
  </div>

  <div style="padding:16px 28px 20px;">
    <div style="padding:12px 14px;background:#f4f4f4;border-left:4px solid {warn_col};
      border-radius:4px;font-size:13px;color:{warn_col};font-weight:bold;">
      {warn_msg}
    </div>
  </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:Arial,sans-serif;">
<div style="max-width:640px;margin:24px auto;background:#fff;border-radius:8px;
  overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

  <div style="background:{NAV};padding:20px 28px;">
    <div style="color:#fff;font-size:18px;font-weight:bold;">EXIT SCANNER</div>
    <div style="color:#DCE6F1;font-size:13px;margin-top:4px;">
      {ticker} ${strike:.0f} {direction} &nbsp;&mdash;&nbsp; {contracts} contract{'s' if contracts != 1 else ''}</div>
    <div style="color:#9ab5cc;font-size:11px;margin-top:2px;">Checked at {time_str}</div>
  </div>

  {body}

  <div style="background:#f4f4f4;padding:12px 28px;font-size:11px;color:#888;
    text-align:center;border-top:1px solid #e0e0e0;margin-top:16px;">
    Generated by Claude &nbsp;|&nbsp; For informational purposes only — not financial advice
  </div>
</div></body></html>"""

# ── EMAIL ─────────────────────────────────────────────────────────────────
def send_email(subject, html):
    if not GMAIL_APP_PASSWORD:
        print("No GMAIL_APP_PASSWORD — skipping email.")
        return
    msg = MIMEMultipart("alternative")
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(EMAIL_FROM, GMAIL_APP_PASSWORD)
        s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}  Emailed to {EMAIL_TO}")

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    if len(args) < 4:
        print("Usage: python3 exit_scanner.py TICKER STRIKE CALL/PUT ENTRY_PRICE [CONTRACTS]")
        print("  e.g. python3 exit_scanner.py SPY 750 PUT 2.27 5")
        sys.exit(1)

    ticker    = args[0].upper()
    strike    = float(args[1])
    direction = args[2].upper()
    entry     = float(args[3])
    contracts = int(args[4]) if len(args) > 4 else 1

    print(f"\nChecking {ticker} ${strike:.0f} {direction}  entry ${entry:.2f}  {contracts} contract{'s' if contracts != 1 else ''}...")

    opt                  = get_option(ticker, strike, direction)
    u_price, u_pct       = get_underlying(ticker)

    if opt and "error" not in opt:
        mid       = opt["mid"]
        pnl_pct   = round((mid - entry) / entry * 100, 1) if entry else 0
        pnl_total = round((mid - entry) * 100 * contracts, 2)
        arrow     = "▲" if pnl_total >= 0 else "▼"
        print(f"  Current: ${mid:.2f}  |  P&L: {arrow} ${abs(pnl_total):.2f} ({pnl_pct:+.1f}%)")
        subject = (f"\U0001f4ca Exit Check: {ticker} ${strike:.0f} {direction} — "
                   f"{arrow} ${abs(pnl_total):.2f} ({pnl_pct:+.1f}%) — "
                   f"{datetime.now().strftime('%I:%M %p CT')}")
    else:
        print(f"  Error: {opt}")
        subject = f"\U0001f4ca Exit Check: {ticker} ${strike:.0f} {direction} — Error"

    html = build_html(ticker, strike, direction, entry, contracts, opt, u_price or 0, u_pct or 0)
    send_email(subject, html)

if __name__ == "__main__":
    main()
