"""
Shared helpers for the Bloom Energy Streamlit app — used by both the
Spread Calculator (home page) and the Options Calculator (pages/).

Nothing in here renders any Streamlit UI; it's imported by both pages,
each of which builds its own independent interface on top of these.
"""

import re
from datetime import datetime

import requests
import streamlit as st

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient

OSI_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


@st.cache_resource
def get_clients():
    api_key = st.secrets["ALPACA_API_KEY"]
    secret_key = st.secrets["ALPACA_SECRET_KEY"]
    option_client = OptionHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    stock_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    return option_client, stock_client


def fetch_with_retry(fetch_function, max_retries=3, delay_seconds=5):
    """Retries an Alpaca call with a delay if rate-limited, instead of failing immediately."""
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


def parse_osi_symbol(osi_symbol: str):
    """Parse an OSI-format option symbol into (underlying, expiration, type, strike)."""
    match = OSI_RE.match(osi_symbol)
    if not match:
        return None
    underlying, exp_str, cp, strike_str = match.groups()
    expiration_date = datetime.strptime(exp_str, "%y%m%d").date()
    option_type = "call" if cp == "C" else "put"
    strike = int(strike_str) / 1000.0
    return underlying, expiration_date, option_type, strike


def get_mid_price(bid, ask, last, max_spread_pct=0.15, min_spread_dollars=0.15):
    """
    Returns (mid_price, is_reliable, reason).

    A quote is flagged unreliable when:
      - the bid-ask spread is wider than `max_spread_pct` of the midpoint
        AND wider than `min_spread_dollars` in absolute terms. Both
        conditions are required together — a cheap, near-expiration
        contract can show a huge PERCENTAGE spread purely from a fixed
        minimum tick width (e.g. $0.35 wide on a $0.50 option = 70%),
        without actually being an illiquid/untrustworthy quote. The
        dollar floor keeps that case from being flagged while still
        catching genuinely wide spreads on higher-priced contracts.
      - there's no usable bid/ask at all and the price falls back to the
        last trade, which carries no guarantee of being recent.
      - there's no usable price at all (returns 0.0, unreliable).

    This doesn't fix bad data — Alpaca is still the source of a wide or
    stale quote either way. It makes that unreliability visible to the
    caller instead of silently blending it into the mid price as if it
    were a tight, trustworthy quote.
    """
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        mid = (bid + ask) / 2
        spread_dollars = ask - bid
        spread_pct = spread_dollars / mid if mid > 0 else 1.0
        if spread_pct <= max_spread_pct or spread_dollars < min_spread_dollars:
            return round(mid, 2), True, None
        return round(mid, 2), False, f"wide bid-ask spread ({spread_pct * 100:.0f}%, ${spread_dollars:.2f})"

    if last is not None and last > 0:
        return round(last, 2), False, "no live bid/ask, fell back to last trade"

    return 0.0, False, "no usable bid, ask, or last price"


@st.cache_data(ttl=3600)
def fetch_risk_free_rate() -> float | None:
    """
    Pull the latest 1-Month Treasury Constant Maturity Rate (FRED series
    DGS1MO) from FRED's public CSV endpoint. No API key required.

    Alpaca does not expose any risk-free-rate or treasury-yield data
    itself — their only "interest rate" surface is the High-Yield Cash
    program paid on idle account balances, unrelated to options pricing
    — so this has to come from an external source either way.

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