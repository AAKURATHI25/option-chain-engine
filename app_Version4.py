import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Option Chain Engine", layout="wide")
st.title("Option Chain Sentiment & Value Engine")

uploaded = st.file_uploader("Upload NSE Option Chain CSV", type=["csv"])

def clean_option_chain(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().upper() for c in df.columns]

    cols = {}
    for c in df.columns:
        if "STRIKE" in c:
            cols[c] = "strike"
        elif "OI" in c and ("CALL" in c or "CE" in c):
            cols[c] = "call_oi"
        elif "OI" in c and ("PUT" in c or "PE" in c):
            cols[c] = "put_oi"
        elif ("LTP" in c or "LAST TRADED PRICE" in c) and ("CALL" in c or "CE" in c):
            cols[c] = "call_ltp"
        elif ("LTP" in c or "LAST TRADED PRICE" in c) and ("PUT" in c or "PE" in c):
            cols[c] = "put_ltp"

    df = df.rename(columns=cols)

    needed = ["strike", "call_oi", "put_oi", "call_ltp", "put_ltp"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    df = df[needed].copy()

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    df[["call_oi","put_oi","call_ltp","put_ltp"]] = df[["call_oi","put_oi","call_ltp","put_ltp"]].fillna(0)

    return df

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
    raw = pd.read_csv(uploaded)
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"{spot:.2f}")
    c2.metric("ATM Strike", f"{atm:.2f}")
    c3.metric("Max Pain", f"{max_pain_strike:.2f}")
    c4.metric("PCR", f"{pcr:.3f}" if pd.notna(pcr) else "NA")

    st.subheader("Computed Table")
    st.dataframe(m, use_container_width=True)

    st.subheader("Time Value vs Strike")
    tv_plot = m.melt(id_vars="strike", value_vars=["call_tv", "put_tv"], var_name="type", value_name="time_value")
    fig_tv = px.area(tv_plot, x="strike", y="time_value", color="type")
    st.plotly_chart(fig_tv, use_container_width=True)

    st.subheader("Open Interest by Strike")
    oi_plot = m.melt(id_vars="strike", value_vars=["call_oi", "put_oi"], var_name="type", value_name="oi")
    fig_oi = px.bar(oi_plot, x="strike", y="oi", color="type", barmode="group")
    st.plotly_chart(fig_oi, use_container_width=True)

    st.subheader("Max Pain Curve")
    fig_mp = px.line(mp_curve, x="strike", y="total_payout")
    st.plotly_chart(fig_mp, use_container_width=True)
else:
    st.info("Upload an NSE option chain CSV to begin.")
