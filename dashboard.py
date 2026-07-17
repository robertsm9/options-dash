import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


# =========================================================
# CONFIGURATION
# =========================================================

TICKER_SYMBOL = "BE"

SPREADS = {
    "230/260 Centered Spread": {
        "long_strike": 230.0,
        "short_strike": 260.0
    }
}

SHARE_PRICE_SCENARIOS = [
    250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350,
    360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460,
    470, 480, 490, 500, 600, 700
]


# =========================================================
# FUNCTIONS
# =========================================================

def get_mid_price(row):
    """
    Return the midpoint between bid and ask.

    If valid bid/ask prices are unavailable, use the last traded price.
    """

    bid = row["bid"]
    ask = row["ask"]
    last = row["lastPrice"]

    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return round((bid + ask) / 2, 2)

    if pd.notna(last):
        return round(last, 2)

    return 0.0


def get_option_row(calls_df, target_strike):
    """
    Find the row matching the requested strike.

    Yahoo normally returns exact strike values, but np.isclose helps
    prevent floating-point matching problems.
    """

    matching_rows = calls_df[
        np.isclose(
            calls_df["strike"].astype(float),
            float(target_strike),
            atol=0.001
        )
    ]

    if matching_rows.empty:
        return None

    return matching_rows.iloc[0]


@st.cache_data(ttl=1800)
def fetch_options_data(
    ticker_symbol,
    selected_expiry,
    long_strike,
    short_strike
):
    """
    Pull the current Yahoo Finance option chain for one expiration.
    """

    ticker = yf.Ticker(ticker_symbol)
    chain = ticker.option_chain(selected_expiry)

    calls_df = chain.calls.copy()
    calls_df["mid"] = calls_df.apply(get_mid_price, axis=1)

    long_row = get_option_row(calls_df, long_strike)
    short_row = get_option_row(calls_df, short_strike)

    if long_row is None:
        raise ValueError(
            f"The {long_strike:.0f} long-call strike was not found "
            f"for expiration {selected_expiry}."
        )

    if short_row is None:
        raise ValueError(
            f"The {short_strike:.0f} short-call strike was not found "
            f"for expiration {selected_expiry}."
        )

    long_premium = float(long_row["mid"])
    short_premium = float(short_row["mid"])

    long_iv = float(long_row["impliedVolatility"])
    short_iv = float(short_row["impliedVolatility"])

    history = ticker.history(period="5d")

    if history.empty:
        raise ValueError(
            f"Yahoo Finance did not return a current price for "
            f"{ticker_symbol}."
        )

    stock_price = float(history["Close"].dropna().iloc[-1])

    expiry_dt = datetime.strptime(selected_expiry, "%Y-%m-%d")
    current_dte = max(
        0,
        (expiry_dt.date() - datetime.now().date()).days
    )

    available_strikes = sorted(
        calls_df["strike"].astype(float).unique().tolist()
    )

    return {
        "long_premium": round(long_premium, 2),
        "short_premium": round(short_premium, 2),
        "long_iv": long_iv,
        "short_iv": short_iv,
        "stock_price": round(stock_price, 2),
        "current_dte": current_dte,
        "available_strikes": available_strikes
    }


@st.cache_data(ttl=1800)
def fetch_spread_term_structure(
    ticker_symbol,
    long_strike,
    short_strike
):
    """
    Pull the current prices of the same call spread across every
    available Yahoo Finance expiration.

    Every row represents a different real option contract priced now.
    """

    ticker = yf.Ticker(ticker_symbol)
    available_expirations = list(ticker.options)

    if not available_expirations:
        raise ValueError(
            f"No option expirations were found for {ticker_symbol}."
        )

    today = datetime.now().date()
    term_rows = []

    for expiration in available_expirations:
        try:
            expiry_date = datetime.strptime(
                expiration,
                "%Y-%m-%d"
            ).date()

            dte = (expiry_date - today).days

            if dte < 0:
                continue

            chain = ticker.option_chain(expiration)
            calls_df = chain.calls.copy()
            calls_df["mid"] = calls_df.apply(
                get_mid_price,
                axis=1
            )

            long_row = get_option_row(
                calls_df,
                long_strike
            )

            short_row = get_option_row(
                calls_df,
                short_strike
            )

            # Skip expirations where one of the strikes does not exist.
            if long_row is None or short_row is None:
                continue

            long_premium = float(long_row["mid"])
            short_premium = float(short_row["mid"])

            # Skip invalid or unusable quotes.
            if long_premium <= 0 or short_premium < 0:
                continue

            net_debit = long_premium - short_premium

            # A normal debit bull-call spread should not have
            # a zero or negative debit.
            if net_debit <= 0:
                continue

            spread_width = short_strike - long_strike
            max_profit = spread_width - net_debit
            max_loss = net_debit
            breakeven = long_strike + net_debit

            long_iv = float(long_row["impliedVolatility"])
            short_iv = float(short_row["impliedVolatility"])

            term_rows.append({
                "Expiration": expiration,
                "Expiration Date": expiry_date,
                "DTE": dte,
                "Long Call Premium": round(
                    long_premium,
                    2
                ),
                "Short Call Premium": round(
                    short_premium,
                    2
                ),
                "Net Debit": round(
                    net_debit,
                    2
                ),
                "Cost per Spread": round(
                    net_debit * 100,
                    2
                ),
                "Max Profit per Share": round(
                    max_profit,
                    2
                ),
                "Max Loss per Share": round(
                    max_loss,
                    2
                ),
                "Breakeven": round(
                    breakeven,
                    2
                ),
                "Long IV": long_iv,
                "Short IV": short_iv
            })

        except Exception:
            # One bad or unavailable expiration should not stop
            # the entire dashboard.
            continue

    if not term_rows:
        raise ValueError(
            f"Yahoo Finance did not return usable {long_strike:.0f}/"
            f"{short_strike:.0f} spread quotes for any expiration."
        )

    term_df = pd.DataFrame(term_rows)
    term_df = term_df.sort_values("DTE").reset_index(drop=True)

    return term_df


def select_closest_expirations(
    term_df,
    target_dte,
    number_of_expirations=3
):
    """
    Select real Yahoo expirations closest to the chosen DTE.

    This does not create synthetic 20-DTE or 0-DTE contracts.
    It only returns contracts that Yahoo currently lists.
    """

    available = term_df.copy()

    available["Distance From Target"] = (
        available["DTE"] - int(target_dte)
    ).abs()

    selected = (
        available
        .sort_values(
            ["Distance From Target", "DTE"]
        )
        .head(number_of_expirations)
        .sort_values("DTE", ascending=False)
        .reset_index(drop=True)
    )

    return selected


def calculate_bull_call_spread(
    long_strike,
    short_strike,
    long_premium,
    short_premium,
    num_spreads,
    actual_dte
):
    """
    Calculate expiration payoff values for one selected spread.
    """

    net_debit = long_premium - short_premium
    spread_width = short_strike - long_strike

    max_profit = spread_width - net_debit
    max_loss = net_debit
    breakeven = long_strike + net_debit

    rows = []

    for share_price in SHARE_PRICE_SCENARIOS:
        long_call_value = max(
            share_price - long_strike,
            0
        )

        short_call_value = -max(
            share_price - short_strike,
            0
        )

        profit_per_spread = (
            long_call_value
            + short_call_value
            - net_debit
        ) * 100

        total_profit = (
            profit_per_spread
            * num_spreads
        )

        max_loss_per_contract = (
            max_loss * 100
        )

        return_pct = (
            profit_per_spread
            / max_loss_per_contract
        ) * 100 if max_loss_per_contract > 0 else 0

        return_per_dte = (
            return_pct / actual_dte
        ) if actual_dte > 0 else 0

        rows.append({
            "Share Price": share_price,
            "Call Spread Cost": round(
                net_debit * 100,
                2
            ),
            "Call Spreads": num_spreads,
            "Profit per Spread": round(
                profit_per_spread,
                2
            ),
            "Total Profit": round(
                total_profit,
                2
            ),
            "Return %": round(
                return_pct,
                2
            ),
            "Return/DTE %": round(
                return_per_dte,
                2
            )
        })

    return {
        "net_debit": round(net_debit, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "breakeven": round(breakeven, 2),
        "return_on_risk": round(
            (max_profit / max_loss) * 100,
            2
        ) if max_loss > 0 else 0,
        "table": pd.DataFrame(rows)
    }


def calculate_expiration_pnl(
    share_price,
    long_strike,
    short_strike,
    net_debit,
    num_spreads
):
    """
    Calculate exact expiration P&L using the current market debit
    for a particular expiration.
    """

    long_payoff = max(
        share_price - long_strike,
        0
    )

    short_payoff = -max(
        share_price - short_strike,
        0
    )

    pnl_per_share = (
        long_payoff
        + short_payoff
        - net_debit
    )

    total_pnl = (
        pnl_per_share
        * 100
        * num_spreads
    )

    return round(total_pnl, 2)


# =========================================================
# DASHBOARD
# =========================================================

st.set_page_config(
    page_title="Bull Call Spread Dashboard",
    layout="wide"
)

st.title("Bull Call Spread Return Dashboard")

col_a, col_b, col_c = st.columns(3)

with col_a:
    ticker_symbol = st.text_input(
        "Ticker",
        TICKER_SYMBOL
    ).upper().strip()


try:
    ticker_obj = yf.Ticker(ticker_symbol)
    available_expirations = ticker_obj.options

    if not available_expirations:
        st.error(
            f"No option chains found for ticker symbol "
            f"{ticker_symbol}."
        )
        st.stop()

    expiry_options = []

    for expiration in available_expirations:
        expiration_date = datetime.strptime(
            expiration,
            "%Y-%m-%d"
        ).date()

        days_out = (
            expiration_date
            - datetime.now().date()
        ).days

        expiry_options.append(
            f"{expiration} ({max(0, days_out)} DTE)"
        )

    with col_b:
        selected_expiry_display = st.selectbox(
            "Select Expiration Date",
            expiry_options,
            index=min(
                3,
                len(expiry_options) - 1
            )
        )

        selected_expiry = (
            selected_expiry_display
            .split(" ")[0]
        )

    with col_c:
        selected_spread_name = st.selectbox(
            "Select Spread",
            list(SPREADS.keys())
        )

    spread = SPREADS[selected_spread_name]

    long_strike = float(
        spread["long_strike"]
    )

    short_strike = float(
        spread["short_strike"]
    )

    num_spreads = st.sidebar.number_input(
        "Number of Call Spreads",
        min_value=1,
        value=10,
        step=1
    )

    selected_data = fetch_options_data(
        ticker_symbol=ticker_symbol,
        selected_expiry=selected_expiry,
        long_strike=long_strike,
        short_strike=short_strike
    )

    long_premium = selected_data[
        "long_premium"
    ]

    short_premium = selected_data[
        "short_premium"
    ]

    long_iv = selected_data[
        "long_iv"
    ]

    short_iv = selected_data[
        "short_iv"
    ]

    stock_price = selected_data[
        "stock_price"
    ]

    actual_dte = selected_data[
        "current_dte"
    ]

    result = calculate_bull_call_spread(
        long_strike=long_strike,
        short_strike=short_strike,
        long_premium=long_premium,
        short_premium=short_premium,
        num_spreads=num_spreads,
        actual_dte=actual_dte
    )

    st.subheader(
        f"{ticker_symbol} | "
        f"{selected_spread_name} | "
        f"Expiry: {selected_expiry} "
        f"({actual_dte} DTE)"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Current Stock Price",
        f"${stock_price:.2f}"
    )

    m2.metric(
        "Long Call Premium",
        f"${long_premium:.2f}"
    )

    m3.metric(
        "Short Call Premium",
        f"${short_premium:.2f}"
    )

    m4.metric(
        "Net Debit / Share",
        f"${result['net_debit']:.2f}"
    )

    m5, m6, m7, m8 = st.columns(4)

    m5.metric(
        "Max Profit / Share",
        f"${result['max_profit']:.2f}"
    )

    m6.metric(
        "Max Loss / Share",
        f"${result['max_loss']:.2f}"
    )

    m7.metric(
        "Breakeven",
        f"${result['breakeven']:.2f}"
    )

    m8.metric(
        "Return on Risk",
        f"{result['return_on_risk']}%"
    )


    # =====================================================
    # LIVE RISK AND SCENARIO MATRIX
    # =====================================================

    st.markdown("---")

    st.markdown(
        "### 🎛️ Live Risk & Scenario Matrix Engine"
    )

    st.write(
        "Each column below represents a different real option "
        "expiration currently listed by Yahoo Finance. The premiums "
        "and net debit are today's market prices for each contract."
    )

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        custom_spot = st.number_input(
            "Tweak Base Share Price ($)",
            min_value=1.0,
            value=float(stock_price),
            step=1.0
        )

    with col_p2:
        custom_dte = st.slider(
            "Tweak Target Days to Expiration",
            min_value=1,
            max_value=max(
                int(
                    (
                        datetime.strptime(
                            available_expirations[-1],
                            "%Y-%m-%d"
                        ).date()
                        - datetime.now().date()
                    ).days
                ),
                1
            ),
            value=max(
                int(actual_dte),
                1
            )
        )

    # Pull the current price of this spread across
    # all real Yahoo expirations.
    term_structure_df = fetch_spread_term_structure(
        ticker_symbol=ticker_symbol,
        long_strike=long_strike,
        short_strike=short_strike
    )

    # Choose the three real expirations closest to the
    # DTE selected by the user.
    selected_expirations_df = select_closest_expirations(
        term_df=term_structure_df,
        target_dte=custom_dte,
        number_of_expirations=3
    )

    # Preserve the same five price rows from the original layout.
    sim_prices = [
        custom_spot * factor
        for factor in [
             0.80,
            0.85,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.15,
            1.20,
            1.25,
            1.30
        ]
    ]

    matrix_rows = []

    for simulated_price in sim_prices:
        row_data = []

        for _, expiration_row in selected_expirations_df.iterrows():
            expiration_net_debit = float(
                expiration_row["Net Debit"]
            )

            total_pnl = calculate_expiration_pnl(
                share_price=simulated_price,
                long_strike=long_strike,
                short_strike=short_strike,
                net_debit=expiration_net_debit,
                num_spreads=num_spreads
            )

            row_data.append(total_pnl)

        matrix_rows.append(row_data)

    matrix_columns = []

    for _, expiration_row in selected_expirations_df.iterrows():
        expiration_label = (
            f"{expiration_row['Expiration']} "
            f"({int(expiration_row['DTE'])} DTE)"
        )

        matrix_columns.append(
            expiration_label
        )

    df_matrix = pd.DataFrame(
        matrix_rows,
        columns=matrix_columns,
        index=[
            f"${price:,.2f}"
            for price in sim_prices
        ]
    )

    def apply_barchart_green(value):
        if value > 0:
            return (
                "background-color: #e2f0d9; "
                "color: #1f4e37;"
            )

        return (
            "background-color: #fce4d6; "
            "color: #c00000;"
        )

    styled_matrix = (
        df_matrix.style
        .applymap(apply_barchart_green)
        .format("${:,.2f}")
    )

    st.dataframe(
        styled_matrix,
        use_container_width=True
    )

    st.caption(
        f"Matrix values show expiration P&L across "
        f"{num_spreads} spreads. Each expiration uses the "
        f"current Yahoo Finance net debit for that specific contract."
    )


    # =====================================================
    # CURRENT SPREAD PRICING BY EXPIRATION
    # =====================================================

    with st.expander(
        "Show Current Yahoo Pricing for Selected Expirations"
    ):
        display_term_df = selected_expirations_df[
            [
                "Expiration",
                "DTE",
                "Long Call Premium",
                "Short Call Premium",
                "Net Debit",
                "Cost per Spread",
                "Breakeven"
            ]
        ].copy()

        display_term_df["Total Position Cost"] = (
            display_term_df["Cost per Spread"]
            * num_spreads
        )

        styled_term_df = (
            display_term_df.style
            .format({
                "Long Call Premium": "${:,.2f}",
                "Short Call Premium": "${:,.2f}",
                "Net Debit": "${:,.2f}",
                "Cost per Spread": "${:,.2f}",
                "Breakeven": "${:,.2f}",
                "Total Position Cost": "${:,.2f}"
            })
        )

        st.dataframe(
            styled_term_df,
            use_container_width=True
        )


    # =====================================================
    # RETURN TABLE
    # =====================================================

    st.markdown("---")

    st.markdown(
        "### Return Table by Share Price "
        "(At Expiration)"
    )

    styled_return_table = (
        result["table"].style
        .applymap(
            apply_barchart_green,
            subset=[
                "Total Profit",
                "Profit per Spread"
            ]
        )
        .format({
            "Share Price": "${:,.2f}",
            "Call Spread Cost": "${:,.2f}",
            "Profit per Spread": "${:,.2f}",
            "Total Profit": "${:,.2f}",
            "Return %": "{:,.2f}%",
            "Return/DTE %": "{:,.2f}%"
        })
    )

    st.dataframe(
        styled_return_table,
        use_container_width=True
    )


    # =====================================================
    # PAYOFF PROFILE CHART
    # =====================================================

    st.markdown(
        "### Payoff Chart"
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        result["table"]["Share Price"],
        result["table"]["Total Profit"],
        color="#2b5c8f",
        linewidth=2
    )

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        alpha=0.7
    )

    ax.axvline(
        result["breakeven"],
        color="red",
        linestyle=":",
        label=(
            f"Breakeven "
            f"(${result['breakeven']})"
        )
    )

    ax.set_xlabel(
        "Share Price at Expiration"
    )

    ax.set_ylabel(
        "Total P/L Across Position ($)"
    )

    ax.set_title(
        f"{selected_spread_name} "
        f"Total Portfolio Payoff Profile"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)


    # =====================================================
    # TIME EXPIRATION RISK CHART
    # =====================================================

    st.markdown("---")

    st.markdown(
        "### ⏳ Time Expiration Risk Explained"
    )

    st.write(
        "Unlike owning regular stock, options have a strict "
        "expiration timeline. This timeline tracks what happens "
        "if a stock drops temporarily but recovers after the "
        "contract closes."
    )

    timeline = [
        "Today",
        "Month 2 (Market Dip)",
        "Month 3 (Your Expiration Date)",
        "Month 4 (Market Recovery)"
    ]

    stock_path = [
        stock_price,
        stock_price - 20,
        stock_price - 20,
        stock_price + 50
    ]

    fig2, ax2 = plt.subplots(
        figsize=(12, 5),
        dpi=100
    )

    fig2.patch.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")

    ax2.plot(
        timeline,
        stock_path,
        color="#737373",
        linestyle="--",
        linewidth=2.5,
        label="Stock Price Trajectory"
    )

    ax2.scatter(
        timeline,
        stock_path,
        color="#2b2b2b",
        s=60,
        zorder=3
    )

    ax2.axhline(
        long_strike,
        color="#377eb8",
        linestyle=":",
        linewidth=1.5
    )

    ax2.text(
        0,
        long_strike + 3,
        f"Your Profit Floor "
        f"(${long_strike:,.0f})",
        color="#377eb8",
        fontsize=9,
        fontweight="medium"
    )

    ax2.axvline(
        "Month 3 (Your Expiration Date)",
        color="#e41a1c",
        linestyle="-",
        linewidth=2.5
    )

    ax2.text(
        "Month 3 (Your Expiration Date)",
        stock_price + 10,
        " EXPIRATION\n DEADLINE\n Position Closes!",
        color="#e41a1c",
        fontweight="bold",
        fontsize=9,
        ha="center",
        bbox=dict(
            facecolor="#fff5f5",
            edgecolor="#fbb4ae",
            boxstyle="round,pad=0.3"
        )
    )

    ax2.set_ylabel(
        "Asset Price ($)",
        fontsize=11,
        fontweight="medium"
    )

    ax2.yaxis.set_major_formatter(
        StrMethodFormatter("${x:,.0f}")
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#e0e0e0")
    ax2.spines["bottom"].set_color("#e0e0e0")

    ax2.tick_params(
        axis="both",
        colors="#636363",
        labelsize=9
    )

    ax2.grid(
        True,
        axis="y",
        linestyle=":",
        color="#f0f0f0",
        alpha=0.5
    )

    st.pyplot(
        fig2,
        clear_figure=True
    )


except Exception as e:
    st.error(
        f"Error: {e}"
    )
