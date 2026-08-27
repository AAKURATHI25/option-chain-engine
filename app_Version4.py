import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Option Chain Engine", layout="wide")
st.title("Option Chain Sentiment & Value Engine")
st.warning("Disclaimer: This analysis is based on option-chain data captured at a specific point in time. Spot price used here is user-assumed/editable and may differ from live market spot. Results (PCR, ATM, IV, and other metrics) can change as market data updates.")

uploaded = st.file_uploader("Upload NSE Option Chain CSV", type=["csv"])

def clean_option_chain(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Expected from your file:
    # call side -> OI, LTP
    # strike    -> STRIKE
    # put side  -> OI.1, LTP.1
    needed_cols = ["STRIKE", "OI", "OI.1", "LTP", "LTP.1"]
    for c in needed_cols:
        if c not in df.columns:
            return pd.DataFrame(columns=["strike", "call_oi", "put_oi", "call_ltp", "put_ltp"])

    out = df[["STRIKE", "OI", "OI.1", "LTP", "LTP.1"]].copy()
    out.columns = ["strike", "call_oi", "put_oi", "call_ltp", "put_ltp"]

    for c in out.columns:
        out[c] = (
            out[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.strip()
            .replace("-", np.nan)
            .replace("", np.nan)
        )
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["strike"])
    out = out[out["strike"] > 0].sort_values("strike").reset_index(drop=True)
    return out
    
def compute_metrics(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    out = df.copy()
    out["call_iv"] = np.maximum(spot - out["strike"], 0)
    out["put_iv"]  = np.maximum(out["strike"] - spot, 0)
    out["call_tv"] = np.maximum(out["call_ltp"] - out["call_iv"], 0)
    out["put_tv"]  = np.maximum(out["put_ltp"] - out["put_iv"], 0)
    return out

def compute_max_pain(df: pd.DataFrame):
    if df.empty or "strike" not in df.columns:
        return np.nan, pd.DataFrame(columns=["strike", "total_payout"])

    work = df.dropna(subset=["strike"]).copy()
    if work.empty:
        return np.nan, pd.DataFrame(columns=["strike", "total_payout"])

    strikes = work["strike"].to_numpy(dtype=float)
    call_oi = work["call_oi"].fillna(0).to_numpy(dtype=float)
    put_oi  = work["put_oi"].fillna(0).to_numpy(dtype=float)

    if len(strikes) == 0:
        return np.nan, pd.DataFrame(columns=["strike", "total_payout"])

    total_payouts = []
    for x in strikes:
        call_pay = np.sum(np.maximum(x - strikes, 0) * call_oi)
        put_pay  = np.sum(np.maximum(strikes - x, 0) * put_oi)
        total_payouts.append(call_pay + put_pay)

    if len(total_payouts) == 0:
        return np.nan, pd.DataFrame(columns=["strike", "total_payout"])

    mp_idx = int(np.argmin(total_payouts))
    return strikes[mp_idx], pd.DataFrame({"strike": strikes, "total_payout": total_payouts})

if uploaded:
    raw = pd.read_csv(uploaded, skiprows=1)
    df = clean_option_chain(raw)

    st.sidebar.header("Inputs")
    spot = st.sidebar.number_input("Spot Price", min_value=0.0, value=float(df["strike"].median()), step=0.5)

    m = compute_metrics(df, spot)
    if m.empty:
        st.error("Could not parse valid option rows from this CSV.")
        st.stop()

    max_pain_strike, mp_curve = compute_max_pain(m)
    atm_idx = (m["strike"] - spot).abs().idxmin()
    atm = float(m.loc[atm_idx, "strike"])
    pcr = (m["put_oi"].sum() / m["call_oi"].sum()) if m["call_oi"].sum() else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Spot", f"{spot:.2f}")
    c2.metric("ATM Strike", f"{atm:.2f}")
    c3.metric("PCR", f"{pcr:.3f}" if pd.notna(pcr) else "NA")

    st.subheader("Computed Table")
    st.dataframe(m, use_container_width=True)

    st.subheader("Time Value Across Strikes")
    tv_plot = m.melt(id_vars="strike", value_vars=["call_tv", "put_tv"], var_name="type", value_name="time_value")
    fig_tv = px.area(tv_plot, x="strike", y="time_value", color="type")
    st.plotly_chart(fig_tv, use_container_width=True)

    st.subheader("Open Interest Concentration Across Strikes")
    oi_plot = m.melt(id_vars="strike", value_vars=["call_oi", "put_oi"], var_name="type", value_name="oi")
    fig_oi = px.bar(oi_plot, x="strike", y="oi", color="type", barmode="group")
    st.plotly_chart(fig_oi, use_container_width=True)
    

    st.subheader("Implied Volatility Skew Across Strikes")
    iv_plot = m.melt(id_vars="strike", value_vars=["call_iv", "put_iv"], var_name="type", value_name="iv")
    fig_iv = px.line(iv_plot, x="strike", y="iv", color="type", markers=True)
    st.plotly_chart(fig_iv, use_container_width=True)

st.subheader("Chart Interpretation (Auto)")

# PCR sentiment
if pd.notna(pcr):
    if pcr > 1.1:
        pcr_view = "Bullish-to-neutral sentiment (higher Put OI vs Call OI)."
    elif pcr < 0.9:
        pcr_view = "Bearish-to-neutral sentiment (higher Call OI vs Put OI)."
    else:
        pcr_view = "Balanced/neutral sentiment zone."
else:
    pcr_view = "PCR not available."

# ATM row
atm_idx = (m["strike"] - spot).abs().idxmin()
atm_row = m.loc[atm_idx]

call_oi_atm = atm_row["call_oi"]
put_oi_atm = atm_row["put_oi"]
call_iv_atm = atm_row["call_iv"] if "call_iv" in m.columns else np.nan
put_iv_atm = atm_row["put_iv"] if "put_iv" in m.columns else np.nan

# OI concentration
top_call_strike = m.loc[m["call_oi"].idxmax(), "strike"]
top_put_strike = m.loc[m["put_oi"].idxmax(), "strike"]

# IV skew view
if pd.notna(call_iv_atm) and pd.notna(put_iv_atm):
    if put_iv_atm > call_iv_atm:
        iv_view = "Put IV is higher than Call IV near ATM (downside protection demand)."
    elif call_iv_atm > put_iv_atm:
        iv_view = "Call IV is higher than Put IV near ATM (upside speculation demand)."
    else:
        iv_view = "Call and Put IV are similar near ATM."
else:
    iv_view = "IV comparison near ATM not available."

st.info(
    f"""
• **PCR ({pcr:.3f}):** {pcr_view}  
• **ATM Strike:** {atm:.2f} | **Spot Assumed:** {spot:.2f}  
• **Highest Call OI concentration:** Strike **{top_call_strike:.2f}**  
• **Highest Put OI concentration:** Strike **{top_put_strike:.2f}**  
• **ATM OI:** Call **{call_oi_atm:,.0f}**, Put **{put_oi_atm:,.0f}**  
• **IV Skew Insight:** {iv_view}  

_Disclaimer: Interpretation is model-based and indicative, using point-in-time uploaded data._
"""
)
else:
    st.info("Upload an NSE option chain CSV to begin.")
