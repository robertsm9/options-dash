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

DTE_CHOICES = [15, 30, 60, 90]

# FIXED: Renamed the label and declared strikes explicitly as floats
SPREADS = {
    "320/350 Centered Spread": {
        "long_strike": 320.0,
        "short_strike": 350.0
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
def find_closest_expiry(ticker_obj, target_days):
    target_date = datetime.now() + timedelta(days=target_days)
    expirations = ticker_obj.options

    if not expirations:
        raise ValueError("No option chains found.")

    dates = [datetime.strptime(exp, "%Y-%m-%d") for exp in expirations]
    closest_date = min(dates, key=lambda d: abs(d - target_date))

    return closest_date.strftime("%Y-%m-%d")


def get_mid_price(row):
    bid = row["bid"]
    ask = row["ask"]
    last = row["lastPrice"]

    if bid > 0 and ask > 0:
        return round((bid + ask) / 2, 2)

    return round(last, 2)


@st.cache_data(ttl=1800)
def fetch_options_data(ticker_symbol, target_dte):
    ticker = yf.Ticker(ticker_symbol)

    expiry = find_closest_expiry(ticker, target_dte)
    chain = ticker.option_chain(expiry)

    calls_df = chain.calls.copy()
    calls_df["mid"] = calls_df.apply(get_mid_price, axis=1)

    price_map = {}

    # FIXED: Map keys stored cleanly as floats to avoid missing lookups
    for _, row in calls_df.iterrows():
        strike = float(row["strike"])
        price_map[strike] = round(row["mid"], 2)

    stock_price = ticker.history(period="1d")["Close"].iloc[-1]

    return expiry, price_map, round(stock_price, 2)


def calculate_bull_call_spread(long_strike, short_strike, long_premium, short_premium):
    net_debit = long_premium - short_premium
    spread_width = short_strike - long_strike

    max_profit = spread_width - net_debit
    max_loss = net_debit
    breakeven = long_strike + net_debit

    rows = []

    for share_price in SHARE_PRICE_SCENARIOS:
        long_call_value = max(share_price - long_strike, 0)
        short_call_value = -max(share_price - short_strike, 0)

        profit = long_call_value + short_call_value - net_debit
        return_pct = profit / max_loss if max_loss > 0 else 0

        rows.append({
            "Share Price": share_price,
            "Long Call Value": round(long_call_value, 2),
            "Short Call Value": round(short_call_value, 2),
            "Net P/L Per Share": round(profit, 2),
            "Net P/L Per Contract": round(profit * 100, 2),
            "Return on Risk %": round(return_pct * 100, 2)
        })

    return {
        "net_debit": round(net_debit, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "breakeven": round(breakeven, 2),
        "return_on_risk": round((max_profit / max_loss) * 100, 2) if max_loss > 0 else 0,
        "table": pd.DataFrame(rows)
    }


# =========================================================
# DASHBOARD
# =========================================================
st.set_page_config(page_title="Bull Call Spread Dashboard", layout="wide")

st.title("Bull Call Spread Return Dashboard")

col_a, col_b, col_c = st.columns(3)

with col_a:
    ticker_symbol = st.text_input("Ticker", TICKER_SYMBOL).upper()

with col_b:
    selected_dte = st.selectbox("Select DTE", DTE_CHOICES, index=3)

with col_c:
    selected_spread_name = st.selectbox("Select Spread", list(SPREADS.keys()))

spread = SPREADS[selected_spread_name]

long_strike = float(spread["long_strike"])
short_strike = float(spread["short_strike"])

try:
    expiry, price_map, stock_price = fetch_options_data(ticker_symbol, selected_dte)

    long_premium = price_map.get(long_strike, 0)
    short_premium = price_map.get(short_strike, 0)

    if long_premium == 0 or short_premium == 0:
        st.error("Could not find one of the option strikes in Yahoo Finance.")
        st.write(f"**Long Strike Target:** {long_strike} | **Premium Found:** {long_premium}")
        st.write(f"**Short Strike Target:** {short_strike} | **Premium Found:** {short_premium}")
        
        with st.expander("Show all available strikes for this expiry date"):
            st.write(sorted(list(price_map.keys())))

    else:
        result = calculate_bull_call_spread(
            long_strike,
            short_strike,
            long_premium,
            short_premium
        )

        st.subheader(f"{ticker_symbol} | {selected_spread_name} | {selected_dte} DTE")
        st.write(f"Closest expiration: **{expiry}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Stock Price", stock_price)
        m2.metric("Long Call Premium", long_premium)
        m3.metric("Short Call Premium", short_premium)
        m4.metric("Net Debit", result["net_debit"])

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Max Profit / Share", result["max_profit"])
        m6.metric("Max Loss / Share", result["max_loss"])
        m7.metric("Breakeven", result["breakeven"])
        m8.metric("Return on Risk", f"{result['return_on_risk']}%")

        # --- MOVE 1: RETURN SCENARIO TABLE MOVED FIRST ---
        st.markdown("### Return Table by Share Price")
        st.dataframe(result["table"], use_container_width=True)

        # --- MOVE 2: PAYOFF PROFILE CHART SECOND ---
        st.markdown("### Payoff Chart")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            result["table"]["Share Price"],
            result["table"]["Net P/L Per Contract"],
            color="#2b5c8f", 
            linewidth=2
        )
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.axvline(result["breakeven"], color="red", linestyle=":", label=f"Breakeven (${result['breakeven']})")
        ax.set_xlabel("Share Price at Expiration")
        ax.set_ylabel("P/L Per Contract ($)")
        ax.set_title(f"{selected_spread_name} Payoff Profile")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # --- MOVE 3: GRAPH 2 INTEGRATED INSIDE THE MAIN ELSE STATEMENT ---
        st.markdown("---")
        st.markdown("### ⏳ Time Expiration Risk Explained")
        st.write("Unlike owning regular stock, options have a strict expiration timeline. This timeline tracks what happens if a stock drops temporarily but recovers *after* our contract closes.")

        # Simplified historical mockup path
        timeline = ["Today", "Month 2 (Market Dip)", "Month 3 (Your Expiration Date)", "Month 4 (Market Recovery)"]
        stock_path = [stock_price, stock_price - 20, stock_price - 20, stock_price + 50]
            
        fig2, ax2 = plt.subplots(figsize=(12, 5), dpi=100)
        fig2.patch.set_facecolor('#ffffff')
        ax2.set_facecolor('#ffffff')

        # Draw the trajectory line
        ax2.plot(timeline, stock_path, color="#737373", linestyle="--", linewidth=2.5, label="Stock Price Trajectory")
        ax2.scatter(timeline, stock_path, color="#2b2b2b", s=60, zorder=3)

        # Target entry benchmark line
        ax2.axhline(long_strike, color="#377eb8", linestyle=":", linewidth=1.5)
        ax2.text(0, long_strike + 3, f"Your Profit Floor (${long_strike:,.0f})", color="#377eb8", fontsize=9, fontweight='medium')

        # Expiration line marker
        ax2.axvline("Month 3 (Your Expiration Date)", color="#e41a1c", linestyle="-", linewidth=2.5)
        ax2.text("Month 3 (Your Expiration Date)", stock_price + 10, " EXPIRATION\n DEADLINE\n Position Closes!", 
                 color="#e41a1c", fontweight='bold', fontsize=9, ha='center', 
                 bbox=dict(facecolor='#fff5f5', edgecolor='#fbb4ae', boxstyle='round,pad=0.3'))

        # Annotations
        ax2.annotate('Contract Terminates Here\n(Spread loses value due to deadline)', 
                     xy=('Month 3 (Your Expiration Date)', stock_price - 20), 
                     xytext=('Today', stock_price - 45),
                     arrowprops=dict(facecolor='#e41a1c', shrink=0.08, width=1, headwidth=6),
                     fontsize=9, color='#e41a1c', fontweight='medium')

        ax2.annotate('The "Too Late" Rally\n(Only physical stock owners profit)', 
                     xy=('Month 4 (Market Recovery)', stock_price + 50), 
                     xytext=('Month 2 (Market Dip)', stock_price + 35),
                     arrowprops=dict(facecolor='#4daf4a', shrink=0.08, width=1, headwidth=6),
                     fontsize=9, color='#4daf4a', fontweight='medium')

        # Layout styling
        ax2.set_ylabel("Asset Price ($)", fontsize=11, fontweight='medium')
        ax2.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
            
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#e0e0e0')
        ax2.spines['bottom'].set_color('#e0e0e0')
        ax2.tick_params(axis='both', colors='#636363', labelsize=9)
        ax2.grid(True, axis='y', linestyle=':', color='#f0f0f0', alpha=0.5)
            
        st.pyplot(fig2, clear_figure=True)

except Exception as e:
    st.error(f"Error: {e}")