"""
Shared helpers for the Bloom Energy Streamlit app — used by both the
Spread Calculator (home page) and the Options Calculator (pages/).

Migrated from Alpaca to the Public.com brokerage API. Nothing in here
renders any Streamlit UI; it's imported by both pages, each of which
builds its own independent interface on top of these.

============================================================
CONFIRM BEFORE FIRST RUN — see inline "CONFIRM:" comments below
============================================================
Two things could not be verified from Public's publicly indexed docs
(their full interactive reference + Postman collection require being
logged into their developer portal, which isn't reachable from here):

1. BASE_URL below is inferred (https://api.public.com), not confirmed.
   Check your Postman collection or the "Try it" panel in their docs
   once logged in, and correct it if wrong.
2. The exact response shape of the option-expirations endpoint is a
   best guess (expects {"expirationDates": [...]}). If it 400s or
   KeyErrors, print the raw response and adjust `fetch_option_expirations`
   to match what actually comes back.

Everything else here (auth token exchange, get-quotes, get-option-chain)
is built directly from Public's documented request/response schemas.

Setup:
    .streamlit/secrets.toml:
        PUBLIC_SECRET_TOKEN = "..."   # generated once from Public's
                                       # account settings page
        PUBLIC_ACCOUNT_ID = "..."     # from the "Get accounts" endpoint,
                                       # or your account settings page
                                       # (use the raw ID, no "#" prefix)
"""

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import streamlit as st

# CONFIRM: base domain inferred, not verified against Public's live docs.
BASE_URL = "https://api.public.com"

OSI_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


# --------------------------------------------------------------------------
# Auth: Public uses a two-step token exchange, unlike Alpaca's static
# API key/secret. A long-lived Secret Token (generated once, manually,
# from Public's account settings page) is exchanged for a short-lived
# Access Token (JWT) on each session. Access tokens default to 15
# minutes validity, so this is cached for slightly less than that to
# guarantee a fresh token before every use.
# --------------------------------------------------------------------------

@st.cache_data(ttl=600)  # refresh well before the ~15 min token expiry
def get_access_token() -> str:
    secret = st.secrets["PUBLIC_SECRET_TOKEN"]
    resp = requests.post(
        f"{BASE_URL}/userapiauthservice/personal/access-tokens",
        json={"secret": secret, "validityInMinutes": 15},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["accessToken"]


def get_clients():
    """
    Returns (access_token, account_id) — the two things every other
    fetch function in this module needs. Kept as a single function
    named `get_clients()` so spread-mech.py and Options_Calculator.py
    don't need their call sites restructured, even though what's
    returned is conceptually different from Alpaca's SDK client objects.
    """
    access_token = get_access_token()
    account_id = str(st.secrets["PUBLIC_ACCOUNT_ID"]).lstrip("#")
    return access_token, account_id


def _auth_headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}


def fetch_with_retry(fetch_function, max_retries=3, delay_seconds=5):
    """Retries a Public API call with a delay if rate-limited, instead of failing immediately."""
    import time

    last_error = None
    for attempt in range(max_retries):
        try:
            return fetch_function()
        except Exception as e:
            last_error = e
            error_text = str(e).lower()
            if "too many requests" in error_text or "rate limit" in error_text or "429" in error_text:
                time.sleep(delay_seconds)
                continue
            else:
                raise
    raise last_error


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------

def fetch_stock_quote(access_token: str, account_id: str, symbol: str) -> dict:
    """
    Returns the raw quote dict for an equity symbol from Public's
    /quotes endpoint — includes last, bid, ask, and per-field timestamps.
    """
    resp = requests.post(
        f"{BASE_URL}/userapigateway/marketdata/{account_id}/quotes",
        headers=_auth_headers(access_token),
        json={"instruments": [{"symbol": symbol, "type": "EQUITY"}]},
        timeout=10,
    )
    resp.raise_for_status()
    quotes = resp.json().get("quotes", [])
    if not quotes:
        raise ValueError(f"Public API returned no quote for {symbol}.")
    return quotes[0]


def fetch_option_expirations(access_token: str, account_id: str, symbol: str) -> list:
    """
    Returns a list of expiration date strings (YYYY-MM-DD) available
    for this underlying.

    CONFIRM: response shape assumed as {"expirationDates": [...]} —
    verify against a real response and adjust the parsing below if
    the actual key/shape differs.
    """
    resp = requests.post(
        f"{BASE_URL}/userapigateway/marketdata/{account_id}/option-expirations",
        headers=_auth_headers(access_token),
        json={"instrument": {"symbol": symbol, "type": "EQUITY"}},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("expirationDates", data.get("expirations", []))


def fetch_option_chain_for_expiration(access_token: str, account_id: str, symbol: str, expiration_date: str) -> dict:
    """
    Returns the raw {"baseSymbol": ..., "calls": [...], "puts": [...]}
    response for one expiration date (YYYY-MM-DD string), straight from
    Public's documented schema.
    """
    resp = requests.post(
        f"{BASE_URL}/userapigateway/marketdata/{account_id}/option-chain",
        headers=_auth_headers(access_token),
        json={"instrument": {"symbol": symbol, "type": "EQUITY"}, "expirationDate": expiration_date},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _is_regular_market_hours(now_utc=None) -> bool:
    """
    True only during regular US equity market hours (9:30am-4:00pm ET,
    Mon-Fri). Used to gate the staleness check below — outside market
    hours EVERY quote is legitimately hours old (nothing trades), so
    flagging on staleness there would fire on literally every contract
    and swamp the UI with warnings that carry no real signal. Within
    market hours, an old timestamp is a genuine red flag.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    eastern = now_utc.astimezone(ZoneInfo("America/New_York"))
    if eastern.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = eastern.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = eastern.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= eastern <= market_close


def get_mid_price(bid, ask, last, bid_timestamp=None, ask_timestamp=None,
                   max_spread_pct=0.15, max_staleness_minutes=20):
    """
    Returns (mid_price, is_reliable, reason).

    A quote is flagged unreliable when:
      - the bid-ask spread is wider than `max_spread_pct` of the midpoint
        (default 15%) — a common symptom of thin/stale quotes on
        illiquid, far-dated contracts.
      - DURING REGULAR MARKET HOURS ONLY, Public's own bid/ask timestamps
        show the quote is older than `max_staleness_minutes` — a much
        more direct staleness signal than inferring it from spread width
        alone. This check is skipped outside market hours, since every
        quote is naturally stale then and the check would otherwise fire
        on everything indiscriminately.
      - there's no usable bid/ask at all and the price falls back to the
        last trade, which carries no timestamp guarantee here either.
      - there's no usable price at all (returns 0.0, unreliable).

    This doesn't fix bad data — Public is still the source of a wide or
    stale quote either way. It makes that unreliability visible to the
    caller instead of silently blending it into the mid price as if it
    were a tight, trustworthy quote.
    """
    def _is_stale(ts_str):
        if not ts_str:
            return False
        if not _is_regular_market_hours():
            return False
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            return False
        age = datetime.now(timezone.utc) - ts
        return age > timedelta(minutes=max_staleness_minutes)

    if bid is not None and ask is not None and float(bid) > 0 and float(ask) > 0:
        bid, ask = float(bid), float(ask)
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid > 0 else 1.0

        if _is_stale(bid_timestamp) or _is_stale(ask_timestamp):
            return round(mid, 2), False, f"stale quote (older than {max_staleness_minutes} min)"
        if spread_pct > max_spread_pct:
            return round(mid, 2), False, f"wide bid-ask spread ({spread_pct * 100:.0f}%)"
        return round(mid, 2), True, None

    if last is not None and float(last) > 0:
        return round(float(last), 2), False, "no live bid/ask, fell back to last trade"

    return 0.0, False, "no usable bid, ask, or last price"


@st.cache_data(ttl=3600)
def fetch_risk_free_rate() -> float | None:
    """
    Pull the latest 1-Month Treasury Constant Maturity Rate (FRED series
    DGS1MO) from FRED's public CSV endpoint. No API key required.

    Neither Alpaca nor Public expose a risk-free-rate/treasury-yield
    surface directly, so this stays as an external FRED lookup either way.

    Returns a percentage (e.g. 4.08) or None if unavailable.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": "DGS1MO"}
    try:
        resp = requests.get(
            url, params=params, timeout=5,
            headers={"User-Agent": "Mozilla/5.0 (compatible; options-calc/1.0)"},
        )
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        for line in reversed(lines[1:]):
            parts = line.split(",")
            if len(parts) != 2:
                continue
            _, value_str = parts
            if value_str and value_str != ".":
                return float(value_str)
        return None
    except Exception:
        return None