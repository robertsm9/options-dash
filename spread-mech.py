"""
Bloom Energy Spread Calculator — home page of the multipage app.

The Options Calculator lives as a separate page under pages/, and runs
fully independently (Streamlit only executes the code for whichever
page is currently selected — no shared API calls firing in the
background between them).

Setup:
    pip install streamlit alpaca-py scipy numpy requests openpyxl

    .streamlit/secrets.toml:
        ALPACA_API_KEY = "..."
        ALPACA_SECRET_KEY = "..."

Run:
    streamlit run spread-mech.py
"""

from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import norm

from alpaca.data.requests import OptionChainRequest, StockLatestTradeRequest

from common import (
    fetch_risk_free_rate,
    fetch_with_retry,
    get_clients,
    get_mid_price,
    parse_osi_symbol,
)

TICKER_SYMBOL = "BE"


# =========================================================
# FUNCTIONS
# =========================================================
def get_option_row(calls_df, target_strike, tolerance=2.5):
    """
    Find the contract at target_strike. Tries an exact match first; if
    none exists, snaps to the nearest listed strike within `tolerance`
    dollars and reports the actual strike used, so a small mismatch
    (e.g. typing 260.00 when the real listed strike is 260.50) doesn't
    silently drop that expiration from the results.

    Returns (row_or_None, actual_strike_used_or_None).
    """
    strikes = calls_df["strike"].astype(float)

    exact = calls_df[np.isclose(strikes, float(target_strike), atol=0.001)]
    if not exact.empty:
        return exact.iloc[0], float(target_strike)

    diffs = (strikes - float(target_strike)).abs()
    nearest_idx = diffs.idxmin()
    if diffs.loc[nearest_idx] <= tolerance:
        row = calls_df.loc[nearest_idx]
        return row, float(row["strike"])

    return None, None


def bs_call_price(S, K, T_days, r, sigma, q=0.0):
    T = T_days / 365.0
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K), 0.0, 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price, d1, d2


def bs_call_delta(S, K, T_days, r, sigma, q=0.0):
    T = T_days / 365.0
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(np.exp(-q * T) * norm.cdf(d1))


def implied_vol_call(market_price, S, K, T_days, r, q=0.0):
    """Solve for sigma given an observed call price. Returns a decimal (0.30 = 30%) or None."""
    if market_price is None or market_price <= 0 or T_days <= 0:
        return None

    def objective(sigma):
        price, _, _ = bs_call_price(S, K, T_days, r, sigma, q)
        return price - market_price

    try:
        low, high = objective(1e-4), objective(5.0)
        if low * high > 0:
            return None
        return brentq(objective, 1e-4, 5.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        return None


@st.cache_data(ttl=300)
def fetch_current_price(_stock_client, ticker_symbol):
    request = StockLatestTradeRequest(symbol_or_symbols=ticker_symbol)
    trade = _stock_client.get_stock_latest_trade(request)
    price = trade[ticker_symbol].price
    if price is None:
        raise ValueError(f"Alpaca did not return a current price for {ticker_symbol}.")
    return round(float(price), 2)


@st.cache_data(ttl=300)
def fetch_all_calls(_option_client, ticker_symbol, current_price, risk_free_rate):
    """
    Pull the full option chain (calls only). Per contract: Alpaca's own
    IV/delta first, solved locally from that contract's own market price
    only if Alpaca doesn't supply it. No borrowing from other strikes.
    """
    request = OptionChainRequest(underlying_symbol=ticker_symbol)
    chain = _option_client.get_option_chain(request)

    today = datetime.now().date()
    rows = []

    for osi_symbol, snapshot in chain.items():
        parsed = parse_osi_symbol(osi_symbol)
        if not parsed:
            continue
        underlying, expiration_date, option_type, strike = parsed
        if option_type != "call":
            continue

        last_price = snapshot.latest_trade.price if snapshot.latest_trade else None
        bid = snapshot.latest_quote.bid_price if snapshot.latest_quote else None
        ask = snapshot.latest_quote.ask_price if snapshot.latest_quote else None
        mid = get_mid_price(bid, ask, last_price)

        raw_iv = getattr(snapshot, "implied_volatility", None)
        greeks = getattr(snapshot, "greeks", None)
        raw_delta = getattr(greeks, "delta", None) if greeks is not None else None

        dte = (expiration_date - today).days

        if raw_iv is not None:
            iv_pct = float(raw_iv) * 100
            iv_source = "alpaca"
        else:
            solved_iv = None
            if mid > 0 and dte > 0 and risk_free_rate is not None:
                solved_iv = implied_vol_call(mid, current_price, strike, dte, risk_free_rate / 100.0)
            iv_pct = solved_iv * 100 if solved_iv is not None else None
            iv_source = "solved" if solved_iv is not None else None

        if raw_delta is not None:
            delta_val = float(raw_delta)
            delta_source = "alpaca"
        else:
            delta_val = None
            delta_source = None
            if iv_pct is not None and dte > 0 and risk_free_rate is not None:
                delta_val = bs_call_delta(current_price, strike, dte, risk_free_rate / 100.0, iv_pct / 100.0)
                delta_source = "calculated"

        rows.append({
            "osi_symbol": osi_symbol,
            "expiration": expiration_date,
            "strike": strike,
            "mid": mid,
            "implied_volatility": iv_pct,
            "iv_source": iv_source,
            "delta": delta_val,
            "delta_source": delta_source,
        })

    if not rows:
        raise ValueError(f"No call option data returned for {ticker_symbol}.")

    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_spread_term_structure(
    _option_client, ticker_symbol, long_strike, short_strike, covered_call_strike,
    current_price, risk_free_rate,
):
    all_calls = fetch_all_calls(_option_client, ticker_symbol, current_price, risk_free_rate)

    today = datetime.now().date()
    rows = []

    for expiration_date, group in all_calls.groupby("expiration"):
        dte = (expiration_date - today).days
        if dte < 0:
            continue

        long_row, long_actual_strike = get_option_row(group, long_strike)
        short_row, short_actual_strike = get_option_row(group, short_strike)
        covered_call_row, covered_call_actual_strike = get_option_row(group, covered_call_strike)

        if long_row is None or short_row is None:
            continue

        long_premium = float(long_row["mid"])
        short_premium = float(short_row["mid"])
        if long_premium <= 0 or short_premium < 0:
            continue

        net_debit = long_premium - short_premium
        if net_debit <= 0:
            continue

        short_iv = short_row["implied_volatility"]
        long_iv = long_row["implied_volatility"]
        short_delta = short_row["delta"]
        long_delta = long_row["delta"]
        short_delta_source = short_row["delta_source"]
        long_delta_source = long_row["delta_source"]

        if covered_call_row is not None:
            covered_call_iv = covered_call_row["implied_volatility"]
            covered_call_premium = float(covered_call_row["mid"])
            covered_call_delta = covered_call_row["delta"]
            covered_call_delta_source = covered_call_row["delta_source"]
        else:
            covered_call_iv = short_iv
            covered_call_premium = short_premium
            covered_call_delta = short_delta
            covered_call_delta_source = short_delta_source
            covered_call_actual_strike = short_actual_strike

        rows.append({
            "Expiration": expiration_date.strftime("%Y-%m-%d"),
            "DTE": dte,
            "Call Bought Premium": round(long_premium, 2),
            "Call Sold Premium": round(short_premium, 2),
            "Implied Vol (Hi)": round(short_iv, 1) if short_iv is not None else None,
            "Implied Vol (Lo)": round(long_iv, 1) if long_iv is not None else None,
            "Implied Vol (Covered Call)": round(covered_call_iv, 1) if covered_call_iv is not None else None,
            "Covered Call Premium": round(covered_call_premium, 2),
            "Call Spread Cost": round(net_debit, 2),
            "Delta (Hi)": round(short_delta, 5) if short_delta is not None else None,
            "Delta (Lo)": round(long_delta, 5) if long_delta is not None else None,
            "Delta (Covered Call)": round(covered_call_delta, 5) if covered_call_delta is not None else None,
            "Delta (Hi) Source": short_delta_source,
            "Delta (Lo) Source": long_delta_source,
            "Delta (Covered Call) Source": covered_call_delta_source,
            "Long Strike Used": long_actual_strike,
            "Short Strike Used": short_actual_strike,
            "Covered Call Strike Used": covered_call_actual_strike,
        })

    if not rows:
        raise ValueError(
            f"No usable {long_strike:.0f}/{short_strike:.0f} spread quotes found for any expiration."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("DTE").reset_index(drop=True)
    return df


def build_metrics_df(term_df, current_price, long_strike, short_strike, covered_call_strike):
    """
    Formulas verified directly against the source spreadsheet:
    - Call spreads          = Covered Call Premium / Call spread cost
    - Marginal IV            = ((IV_i * DTE_i) - (IV_prior * DTE_prior)) / (DTE_i - DTE_prior)
    - Total profit           = Call spreads * Profit/spread
    - Underlying share       = Covered call strike (constant)
    - Combined value         = Total profit + Underlying share
    - Return                 = Total profit / Current share price
    - Return/DTE             = Return / DTE
    - Return/Marginal DTE    = (Return_i - Return_prior) / (DTE_i - DTE_prior)

    IMPORTANT: "Call sold" (the premium that funds the spreads) uses
    the COVERED CALL'S premium, not the spread's Hi strike premium.

    Delta section — sourced from Alpaca's server-side greeks when
    available, else calculated locally via Black-Scholes.

    Returns a "tidy" DataFrame: one row per expiration, one column per
    metric, each column a single consistent type (all float, e.g.). This
    is what makes it safe to render via st.dataframe — Streamlit renders
    through Apache Arrow under the hood, which requires each COLUMN to
    have one inferable type. The previous version built this the other
    way around (metrics as rows, then transposed so each expiration
    became a column) — after transposing, every column ended up mixing
    floats, ints, and "12.3%"-style strings, which Arrow couldn't
    reliably type and crashed on. Percent columns here stay as plain
    floats (e.g. 12.3, not "12.3%"); the "%" is added visually via
    st.column_config when displaying, not baked into the data.
    """
    width = short_strike - long_strike

    dtes = term_df["DTE"].tolist()
    ivs = term_df["Implied Vol (Hi)"].tolist()
    call_sold_display_premiums = term_df["Call Sold Premium"].tolist()
    call_sold_premiums = term_df["Covered Call Premium"].tolist()
    spread_costs = term_df["Call Spread Cost"].tolist()

    hi_deltas = term_df["Delta (Hi)"].tolist()
    lo_deltas = term_df["Delta (Lo)"].tolist()
    covered_call_deltas = term_df["Delta (Covered Call)"].tolist()

    marginal_ivs, num_spreads_list, total_profits = [], [], []
    combined_values, returns, returns_per_dte, returns_per_marginal_dte = [], [], [], []
    spread_delta_per_unit, long_call_spread_total_delta = [], []
    covered_call_delta_contributions, total_position_deltas = [], []

    prev_dte = None
    prev_return = None

    for i in range(len(term_df)):
        dte = dtes[i]
        iv_hi = ivs[i]
        proceeds = call_sold_premiums[i]
        cost = spread_costs[i]

        if i == 0 or iv_hi is None:
            marginal_iv = iv_hi
        else:
            prior_iv = ivs[i - 1]
            prior_dte = dtes[i - 1]
            time_gap = dte - prior_dte
            if time_gap > 0 and prior_iv is not None:
                marginal_iv = ((iv_hi * dte) - (prior_iv * prior_dte)) / time_gap
            else:
                marginal_iv = iv_hi
        marginal_ivs.append(round(marginal_iv, 1) if marginal_iv is not None else None)

        num_spreads = proceeds / cost if cost > 0 else 0
        num_spreads_list.append(round(num_spreads, 1))

        total_profit = num_spreads * width
        total_profits.append(round(total_profit, 1))

        combined_value = total_profit + covered_call_strike
        combined_values.append(round(combined_value, 1))

        ret = total_profit / current_price
        returns.append(round(ret * 100, 1))

        ret_per_dte = (ret / dte) if dte > 0 else 0
        returns_per_dte.append(round(ret_per_dte * 100, 1))

        if i == 0:
            ret_per_marginal_dte = ret_per_dte
        else:
            marginal_dte = dte - prev_dte
            ret_per_marginal_dte = (ret - prev_return) / marginal_dte if marginal_dte > 0 else 0
        returns_per_marginal_dte.append(round(ret_per_marginal_dte * 100, 1))

        prev_dte = dte
        prev_return = ret

        hi_delta = hi_deltas[i] if hi_deltas[i] is not None else 0.0
        lo_delta = lo_deltas[i] if lo_deltas[i] is not None else 0.0
        covered_call_delta = covered_call_deltas[i] if covered_call_deltas[i] is not None else 0.0

        one_spread_delta = lo_delta - hi_delta
        total_spread_delta = num_spreads * one_spread_delta
        covered_call_delta_contribution = -covered_call_delta
        total_position_delta = 1.0 + covered_call_delta_contribution + total_spread_delta

        spread_delta_per_unit.append(round(one_spread_delta, 5))
        long_call_spread_total_delta.append(round(total_spread_delta, 4))
        covered_call_delta_contributions.append(round(covered_call_delta_contribution, 4))
        total_position_deltas.append(round(total_position_delta, 4))

    metrics = {
        "Expiration": term_df["Expiration"].tolist(),
        f"Call bought: {long_strike:.0f}": term_df["Call Bought Premium"].tolist(),
        f"Call sold: {short_strike:.0f} (spread Hi leg)": call_sold_display_premiums,
        f"Implied Volatility (Hi): {short_strike:.0f}": ivs,
        "Marginal IV": marginal_ivs,
        "DTE": dtes,
        f"Call sold (Covered Call @ {covered_call_strike:.0f}, funds spreads)": call_sold_premiums,
        "Call spread cost": spread_costs,
        "Call spreads": num_spreads_list,
        "Profit/spread": [width] * len(term_df),
        "Total profit": total_profits,
        "Underlying share": [covered_call_strike] * len(term_df),
        "Combined value": combined_values,
        "Return %": returns,
        "Return/DTE %": returns_per_dte,
        "Return/Marginal DTE %": returns_per_marginal_dte,
        f"Delta - {short_strike:.0f} call (Hi)": hi_deltas,
        f"Delta - {long_strike:.0f} call (Lo)": lo_deltas,
        f"Delta - {covered_call_strike:.0f} call (Covered Call)": covered_call_deltas,
        "Spread Delta (per single spread)": spread_delta_per_unit,
        "Long Call Spread Delta (total position, x spreads held)": long_call_spread_total_delta,
        "Covered Call Delta Contribution": covered_call_delta_contributions,
        "Equity Delta": [1.0] * len(term_df),
        "Total Position Delta": total_position_deltas,
    }

    return pd.DataFrame(metrics)


def transpose_for_excel(metrics_df):
    """
    Transpose the tidy metrics table into the "Excel-style" layout
    (metrics as rows, expirations as columns) for the spreadsheet
    export. Mixed types within a row are fine here — this never goes
    through Arrow/Streamlit, just openpyxl.
    """
    return metrics_df.set_index("Expiration").T


def convert_to_excel_bytes(metrics_df, current_price, ticker_symbol, long_strike, short_strike, covered_call_strike):
    output = BytesIO()
    excel_layout_df = transpose_for_excel(metrics_df)
    metadata_df = pd.DataFrame({
        "Field": ["Ticker", "Current Share Price", "Long Strike", "Short Strike",
                  "Covered Call Strike", "Data Source", "Generated On"],
        "Value": [
            ticker_symbol, current_price, long_strike, short_strike, covered_call_strike,
            "Alpaca (Basic market data, 15-min delayed)",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ],
    })
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        excel_layout_df.to_excel(writer, sheet_name="Term Structure")
        metadata_df.to_excel(writer, sheet_name="Info", index=False)
    return output.getvalue()


# =========================================================
# PAGE
# =========================================================
st.title("Bloom Energy Spread Calculator")

ticker_symbol = TICKER_SYMBOL
option_client, stock_client = get_clients()

# Risk-free rate: fetched silently in the background (FRED, live 1-month
# Treasury yield), used only for local IV/delta fallback math. No UI
# widget — falls back to a reasonable default if the fetch fails, with
# no banner shown.
risk_free_rate = fetch_risk_free_rate()
if risk_free_rate is None:
    risk_free_rate = 4.0

try:
    live_price = fetch_with_retry(lambda: fetch_current_price(stock_client, ticker_symbol))
    st.metric("Live Price (Alpaca)", f"${live_price:.2f}")

    current_share_price = st.number_input(
        "Current Share Price ($)", min_value=0.01, value=float(live_price), step=1.0
    )
    covered_call_strike = st.number_input(
        "Call Sold (Covered Call Strike)", min_value=1.0, value=270.0, step=5.0
    )
    short_strike = st.number_input(
        "Hi (Call Sold Strike)", min_value=1.0, value=270.0, step=5.0
    )
    long_strike = st.number_input(
        "Lo (Call Bought Strike)", min_value=1.0, value=260.0, step=5.0
    )

    if short_strike <= long_strike:
        st.error("Hi strike must be greater than Lo strike.")
        st.stop()

    all_available_expirations = fetch_with_retry(
        lambda: fetch_spread_term_structure(
            option_client, ticker_symbol=ticker_symbol, long_strike=long_strike,
            short_strike=short_strike, covered_call_strike=covered_call_strike,
            current_price=current_share_price, risk_free_rate=risk_free_rate,
        )
    )

    expiration_options = [
        f"{row['Expiration']} ({row['DTE']} DTE)" for _, row in all_available_expirations.iterrows()
    ]
    selected_expiration_labels = st.multiselect(
        "Select Expirations to Display", options=expiration_options, default=expiration_options
    )
    selected_dtes = [int(label.split("(")[1].replace(" DTE)", "")) for label in selected_expiration_labels]

    term_df = all_available_expirations[
        all_available_expirations["DTE"].isin(selected_dtes)
    ].reset_index(drop=True)

    if term_df.empty:
        st.warning("No expirations selected. Choose at least one from the list above.")
        st.stop()

    metrics_df = build_metrics_df(
        term_df=term_df, current_price=current_share_price,
        long_strike=long_strike, short_strike=short_strike, covered_call_strike=covered_call_strike,
    )

    # --- On-screen display: transpose back to the original spreadsheet
    # layout (metrics as rows, expiration dates as columns), formatted
    # as strings so every cell is a consistent type. Built as a brand
    # new DataFrame (not mutated in place) — transposing an all-numeric
    # table locks pandas' underlying columns into a strict float64
    # block, and writing strings back into that block via .loc row
    # assignment raises "Invalid value for dtype float64". Building
    # fresh from plain Python values sidesteps that entirely. The
    # underlying metrics_df stays fully numeric for the Excel export
    # and the summary stats below, so nothing downstream is affected. ---
    excel_layout_display = transpose_for_excel(metrics_df)

    def format_cell(row_label, value):
        if pd.isna(value):
            return ""
        if "%" in row_label:
            return f"{value:.1f}%"
        if row_label == "DTE":
            return str(int(value))
        if "Delta" in row_label:
            return f"{value:.4f}"
        if isinstance(value, float):
            return f"{value:,.2f}"
        return str(value)

    formatted_display = pd.DataFrame(
        {
            col: [
                format_cell(row_label, excel_layout_display.loc[row_label, col])
                for row_label in excel_layout_display.index
            ]
            for col in excel_layout_display.columns
        },
        index=excel_layout_display.index,
    )

    st.markdown("---")
    st.markdown("### Full Term Structure")
    st.dataframe(formatted_display, use_container_width=True)

    alpaca_count = (
        (term_df["Delta (Hi) Source"] == "alpaca").sum()
        + (term_df["Delta (Lo) Source"] == "alpaca").sum()
        + (term_df["Delta (Covered Call) Source"] == "alpaca").sum()
    )
    calculated_count = sum(
        (term_df[col] == "calculated").sum()
        for col in ["Delta (Hi) Source", "Delta (Lo) Source", "Delta (Covered Call) Source"]
    )
    missing_count = sum(
        term_df[col].isna().sum()
        for col in ["Delta (Hi) Source", "Delta (Lo) Source", "Delta (Covered Call) Source"]
    )
    st.caption(
        f"Delta values: {alpaca_count} from Alpaca's server-side greeks, "
        f"{calculated_count} calculated locally via Black-Scholes using that "
        f"same contract's own implied volatility (solved from its own market "
        f"price when Alpaca didn't report IV directly), {missing_count} "
        f"unavailable (no Alpaca data and no market price to solve IV from)."
    )

    # --- Strike-snap notices: tell the user if any leg didn't have an
    # exact listed match and had to snap to the nearest real strike ---
    strike_checks = [
        ("Lo (Call Bought)", long_strike, "Long Strike Used"),
        ("Hi (Call Sold)", short_strike, "Short Strike Used"),
        ("Covered Call", covered_call_strike, "Covered Call Strike Used"),
    ]
    for label, requested, col in strike_checks:
        actual_strikes = term_df[col].dropna().unique()
        mismatched = [s for s in actual_strikes if abs(s - requested) > 0.001]
        if mismatched:
            mismatched_str = ", ".join(f"${s:.2f}" for s in sorted(mismatched))
            st.info(
                f"Note: no contract listed at exactly ${requested:.2f} for the "
                f"{label} leg on some expirations — used the nearest available "
                f"strike instead ({mismatched_str})."
            )

    if covered_call_strike != short_strike:
        st.info(
            f"Note: your covered call strike (${covered_call_strike:.0f}) is "
            f"different from the spread's Hi strike (${short_strike:.0f}). The "
            f"premium used to fund your spreads comes from the covered call, "
            f"and is calculated separately from the spread's own Hi-strike premium."
        )

    st.markdown("---")
    excel_bytes = convert_to_excel_bytes(
        metrics_df=metrics_df, current_price=current_share_price,
        ticker_symbol=ticker_symbol, long_strike=long_strike,
        short_strike=short_strike, covered_call_strike=covered_call_strike,
    )
    st.download_button(
        label="📥 Generate & Download Excel Sheet", data=excel_bytes,
        file_name=f"BE_call_spread_{long_strike:.0f}_{short_strike:.0f}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

except Exception as e:
    st.error(f"Error: {e}\n\nIf this is a rate-limit error from Alpaca, please wait a few minutes and refresh the page.")