"""
Options Calculator — Alpaca-powered, Barchart-style theoretical pricing
tool. Second page of the multipage app; runs fully independently of the
Spread Calculator home page.

Setup:
    pip install streamlit alpaca-py scipy numpy requests --break-system-packages

    .streamlit/secrets.toml:
        ALPACA_API_KEY = "..."
        ALPACA_SECRET_KEY = "..."
"""

from datetime import date

import numpy as np
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import norm

from alpaca.data.requests import OptionChainRequest, StockLatestTradeRequest

from common import fetch_risk_free_rate, get_clients, parse_osi_symbol


# --------------------------------------------------------------------------
# Black-Scholes pricing engine
# --------------------------------------------------------------------------

def bs_price(S, K, T_days, r, sigma, q, option_type="call"):
    """T_days: days to expiration. Returns (price, d1, d2)."""
    T = T_days / 365.0
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        return intrinsic, 0.0, 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price, d1, d2


def bs_greeks(S, K, T_days, r, sigma, q, d1, d2, option_type="call"):
    T = T_days / 365.0
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365.0
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365.0
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100.0

    return delta, gamma, vega, theta, rho


def implied_vol(market_price, S, K, T_days, r, q, option_type="call"):
    """
    Solve for sigma given an observed market price.

    Returns a (sigma_or_None, reason) tuple so callers can distinguish
    *why* a solve failed instead of just getting back None:
      - "ok"               : solved successfully
      - "no_price"         : market_price/T_days invalid (<=0)
      - "below_intrinsic"  : price is less than intrinsic value (bad/stale quote)
      - "implausible_high" : price implies vol above the search ceiling (bad/stale quote)
      - "solver_error"     : brentq itself raised
    """
    if market_price is None or market_price <= 0 or T_days <= 0:
        return None, "no_price"

    def objective(sigma):
        price, _, _ = bs_price(S, K, T_days, r, sigma, q, option_type)
        return price - market_price

    # Search a wide vol range (0.01% to 2000%) before giving up — illiquid,
    # short-dated, or deep OTM contracts can legitimately imply extreme vols.
    low_sigma, high_sigma = 1e-4, 20.0
    low, high = objective(low_sigma), objective(high_sigma)

    if low > 0:
        # Even near-zero vol overprices vs. the quote -> price is below
        # intrinsic value. Almost always a bad/stale print, not a real vol.
        return None, "below_intrinsic"
    if high < 0:
        # Even at 2000% vol the model can't reach this price.
        return None, "implausible_high"

    try:
        return brentq(objective, low_sigma, high_sigma, xtol=1e-6), "ok"
    except (ValueError, RuntimeError):
        return None, "solver_error"


def smoothed_iv(filtered_rows, target_strike, S, T_days, r, q, option_type_key,
                 window=2, price_field="last"):
    """
    Average the solved IV across nearby strikes (target_strike +/- `window`
    strikes on each side, within the same expiration/option type) to smooth
    out single-quote noise — approximates how vendors like Barchart derive
    a default volatility rather than reading it off one contract's quote.

    Returns (smoothed_iv_pct_or_None, details) where details is a list of
    (strike, price_used, solved_iv_pct) for every neighbor that solved
    successfully, so the result can be shown/audited in the UI.
    """
    strikes_sorted = [row["strike"] for row in filtered_rows]
    if target_strike not in strikes_sorted:
        return None, []

    idx = strikes_sorted.index(target_strike)
    lo = max(0, idx - window)
    hi = min(len(filtered_rows), idx + window + 1)
    neighbors = filtered_rows[lo:hi]

    ivs = []
    details = []
    for row in neighbors:
        price = row.get(price_field) or row.get("last") or row.get("bid") or row.get("ask")
        if not price:
            continue
        solved, _ = implied_vol(price, S, row["strike"], T_days, r, q, option_type_key)
        if solved is not None:
            ivs.append(solved)
            details.append((row["strike"], price, solved * 100))

    if not ivs:
        return None, details
    return float(np.mean(ivs)) * 100, details


# --------------------------------------------------------------------------
# Alpaca data helpers
# --------------------------------------------------------------------------

@st.cache_data(ttl=60)
def fetch_underlying_price(_stock_client, symbol: str) -> float:
    request = StockLatestTradeRequest(symbol_or_symbols=symbol)
    trade = _stock_client.get_stock_latest_trade(request)
    return float(trade[symbol].price)


@st.cache_data(ttl=60)
def fetch_chain(_option_client, symbol: str, expiration: str | None = None):
    kwargs = {"underlying_symbol": symbol}
    if expiration:
        kwargs["expiration_date"] = expiration
    request = OptionChainRequest(**kwargs)
    chain = _option_client.get_option_chain(request)

    rows = []
    for osi_symbol, snapshot in chain.items():
        parsed = parse_osi_symbol(osi_symbol)
        if not parsed:
            continue
        underlying, expiration_date, option_type, strike = parsed

        last_price = snapshot.latest_trade.price if snapshot.latest_trade else None
        bid = snapshot.latest_quote.bid_price if snapshot.latest_quote else None
        ask = snapshot.latest_quote.ask_price if snapshot.latest_quote else None

        rows.append({
            "osi_symbol": osi_symbol,
            "expiration": expiration_date,
            "option_type": option_type,
            "strike": strike,
            "last": last_price,
            "bid": bid,
            "ask": ask,
            "iv": getattr(snapshot, "implied_volatility", None),
            "greeks": getattr(snapshot, "greeks", None),
        })
    return rows


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.title("Options Calculator")

option_client, stock_client = get_clients()

symbol = st.text_input("Enter an equity symbol:", value="AAPL").strip().upper()

if not symbol:
    st.stop()

try:
    underlying_price = fetch_underlying_price(stock_client, symbol)
except Exception as e:
    st.error(f"Couldn't fetch price for {symbol}: {e}")
    st.stop()

st.subheader(f"{symbol} — ${underlying_price:,.2f}")

try:
    all_rows = fetch_chain(option_client, symbol)
except Exception as e:
    st.error(f"Couldn't fetch option chain for {symbol}: {e}")
    st.stop()

if not all_rows:
    st.warning(f"No option chain data returned for {symbol}.")
    st.stop()

expirations = sorted({row["expiration"] for row in all_rows})

col1, col2, col3 = st.columns(3)
with col1:
    option_type_choice = st.selectbox("Option Type", ["Call", "Put"])
with col2:
    expiration_choice = st.selectbox(
        "Expiration", expirations, format_func=lambda d: d.strftime("%Y-%m-%d"),
    )
with col3:
    dte = (expiration_choice - date.today()).days
    st.metric("DTE (days)", dte)

option_type_key = "call" if option_type_choice == "Call" else "put"
filtered = [
    row for row in all_rows
    if row["expiration"] == expiration_choice and row["option_type"] == option_type_key
]
filtered.sort(key=lambda r: r["strike"])

strikes = [row["strike"] for row in filtered]
default_idx = min(
    range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price),
) if strikes else 0

strike_choice = st.selectbox(
    "Strike Price", strikes, index=default_idx if strikes else 0,
    format_func=lambda s: f"{s:.2f}",
)

selected = next((r for r in filtered if r["strike"] == strike_choice), None)

st.divider()
st.subheader("Input Parameters")

# Risk-free rate: defaults to FRED's live 1-month Treasury yield, but is
# a visible, editable field (like Barchart's "Risk-free rate%") so it can
# be manually overridden when comparing against other vendors.
fetched_rate = fetch_risk_free_rate()
if fetched_rate is None:
    fetched_rate = 4.0

row1_left, row1_right = st.columns(2)
with row1_left:
    S = st.number_input("Underlying Price", value=float(underlying_price), format="%.2f")
with row1_right:
    K = st.number_input("Strike Price", value=float(strike_choice), format="%.2f")

row2_left, row2_right = st.columns(2)
with row2_left:
    T_days = st.number_input("DTE (days)", value=int(dte), min_value=0)
with row2_right:
    q_pct = st.number_input(
        "Dividend Yield % (equities)", value=0.0, format="%.2f",
        help="Alpaca's option/stock endpoints don't expose dividend yield. "
             "Defaults to 0% — override manually if this contract's underlying pays a dividend.",
    )
q = q_pct / 100.0

row3_left, row3_right = st.columns(2)
with row3_left:
    risk_free_rate = st.number_input(
        "Risk-free rate %", value=float(fetched_rate), format="%.2f",
        help="Defaults to FRED's live 1-month Treasury yield. Override to "
             "match another vendor's rate (e.g. Barchart) for side-by-side comparisons.",
    )
r = risk_free_rate / 100.0

# --- Volatility %: a FIXED input that feeds Theoretical Price. Matches
# Barchart's behavior — this does NOT move when you change the Market
# Option Price radio below. By default it's seeded from Alpaca's reported
# IV, falling back to a solve off the "Last" price. Optionally, it can
# instead be seeded from a smoothed average across nearby strikes, which
# tends to land closer to what vendors like Barchart show as their default.
seed_price = selected["last"] if selected else None
seed_solved_iv, seed_solve_reason = (None, "no_price")
if seed_price:
    seed_solved_iv, seed_solve_reason = implied_vol(seed_price, S, K, T_days, r, q, option_type_key)

use_smoothed = st.checkbox(
    "Use smoothed IV (average across nearby strikes)",
    value=False,
    help="Solves IV independently at the 2 strikes above and below this one "
         "(same expiration/type) and averages them, instead of using a single "
         "contract's quote. Smooths out single-quote noise and tends to land "
         "closer to vendor-default volatilities (e.g. Barchart's).",
)

smooth_result_iv, smooth_details = (None, [])
if use_smoothed:
    smooth_result_iv, smooth_details = smoothed_iv(
        filtered, strike_choice, S, T_days, r, q, option_type_key,
        window=2, price_field="last",
    )

if use_smoothed and smooth_result_iv is not None:
    seeded_iv = smooth_result_iv
    iv_source = f"smoothed average across {len(smooth_details)} nearby strikes"
elif selected and selected.get("iv"):
    seeded_iv = float(selected["iv"]) * 100
    iv_source = "Alpaca (reported IV)"
elif seed_solved_iv is not None:
    seeded_iv = seed_solved_iv * 100
    iv_source = f"solved from Last price ({seed_price:.2f})"
else:
    seeded_iv = None
    iv_source = None

if use_smoothed and smooth_result_iv is None:
    st.caption(
        "⚠️ Couldn't compute a smoothed IV (no solvable quotes among nearby "
        "strikes) — falling back to the default source."
    )

if use_smoothed and smooth_details:
    with st.expander("Smoothed IV: strikes used", expanded=False):
        for strike_val, price_val, iv_val in smooth_details:
            marker = " ← selected" if strike_val == strike_choice else ""
            st.write(f"Strike {strike_val:.2f} · price {price_val:.2f} · IV {iv_val:.2f}%{marker}")
        st.write(f"**Average: {smooth_result_iv:.2f}%**")

SOLVE_REASON_MESSAGES = {
    "no_price": "No usable market price for this contract/source — likely no recent quote or trade.",
    "below_intrinsic": "This price is below intrinsic value for these inputs — likely a stale or bad print.",
    "implausible_high": "This price implies volatility above 2000% — likely a stale or bad print, not a real quote.",
    "solver_error": "The volatility solver failed unexpectedly for these inputs.",
}

if seeded_iv is not None:
    vol_pct = st.number_input(
        "Volatility %", value=float(seeded_iv), format="%.2f", help=f"Source: {iv_source}",
    )
else:
    vol_pct = st.number_input(
        "Volatility %", value=0.0, format="%.2f",
        help="No volatility available. Enter one manually.",
    )
    if seed_price:
        st.caption(f"⚠️ {SOLVE_REASON_MESSAGES.get(seed_solve_reason, 'Could not solve implied volatility.')}")

sigma = vol_pct / 100.0

# --- Debug panel: shows exactly what Alpaca returned, raw, for this
# contract, so a "0.00" downstream can be traced back to either
# "no quote at all" or "quote present but unsolvable". ---
with st.expander("Debug: raw contract data from Alpaca", expanded=False):
    if selected:
        st.write(
            f"osi_symbol: `{selected.get('osi_symbol')}`  \n"
            f"last: `{selected.get('last')}` · bid: `{selected.get('bid')}` · "
            f"ask: `{selected.get('ask')}`  \n"
            f"Alpaca reported IV: `{selected.get('iv')}`"
        )
    else:
        st.write("No contract matched for this expiration/strike/type.")

st.divider()
calc_col, iv_col = st.columns(2)

with calc_col:
    st.markdown("### Calculated Theoretical Values")
    if sigma > 0:
        price, d1, d2 = bs_price(S, K, T_days, r, sigma, q, option_type_key)
        delta, gamma, vega, theta, rho = bs_greeks(S, K, T_days, r, sigma, q, d1, d2, option_type_key)
        st.write(f"**Theoretical Price:** {price:.4f}")
        st.write(f"**Delta:** {delta:.5f}")
        st.write(f"**Gamma:** {gamma:.5f}")
        st.write(f"**Vega:** {vega:.5f}")
        st.write(f"**Theta:** {theta:.5f}")
        st.write(f"**Rho:** {rho:.5f}")
    else:
        st.info("Enter a nonzero volatility above to see theoretical values.")

with iv_col:
    st.markdown("### IV Calculation")
    st.write(f"**Option:** {option_type_choice}")

    # Diagnostic only: pick a quote type and see what vol it implies,
    # compared against the fixed Volatility % benchmark above. Changing
    # this does NOT affect Theoretical Price.
    price_source = st.radio(
        "Market Option Price", ["Last", "Bid", "Ask", "Mid"], horizontal=True,
    )
    price_map = {
        "Last": selected["last"] if selected else None,
        "Bid": selected["bid"] if selected else None,
        "Ask": selected["ask"] if selected else None,
        "Mid": (
            (selected["bid"] + selected["ask"]) / 2
            if selected and selected["bid"] and selected["ask"] else None
        ),
    }
    obs_price = price_map.get(price_source)

    solved_iv, solve_reason = (None, "no_price")
    if obs_price:
        solved_iv, solve_reason = implied_vol(obs_price, S, K, T_days, r, q, option_type_key)

    if obs_price:
        st.write(f"**Market Option Price:** {obs_price:.2f}")
        if solved_iv is not None:
            st.write(f"**Implied Volatility:** {solved_iv * 100:.2f}%")
        else:
            st.write(
                f"**Implied Volatility:** could not solve "
                f"({SOLVE_REASON_MESSAGES.get(solve_reason, 'unknown reason')})"
            )
    else:
        st.write("No market price available for this contract/source.")

    st.markdown("**Alpaca Reported Greeks**")
    if selected and selected.get("greeks"):
        greeks_obj = selected["greeks"]
        greeks_dict = greeks_obj.__dict__ if hasattr(greeks_obj, "__dict__") else greeks_obj
        st.json(greeks_dict)
    else:
        st.caption("No server-side greeks available for this contract right now (typically means no recent quote/trade).")

st.divider()
st.caption(
    "Data: Alpaca Basic market data (15-min delayed) for price, chain, and IV. "
    "Risk-free rate: defaults to FRED's live 1-month Treasury yield, editable above. "
    "Dividend yield: not available via API, defaults to 0% unless overridden. "
    "Volatility %% is a fixed benchmark seeded once from Alpaca's reported IV (or "
    "solved from the Last price) — it does not change when you switch the Market "
    "Option Price radio in IV Calculation, matching Barchart's layout. Theoretical "
    "values computed locally via Black-Scholes using the inputs above — no hidden "
    "defaults. Note: inputs reset to live defaults on every page visit/rerun (no "
    "persistence)."
)