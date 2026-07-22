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


def get_mid_price(bid, ask, last):
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return round((bid + ask) / 2, 2)
    if last is not None and last > 0:
        return round(last, 2)
    return 0.0


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