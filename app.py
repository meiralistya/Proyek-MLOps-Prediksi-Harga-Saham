import streamlit as st
import joblib
import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz

# CONFIG
st.set_page_config(
    page_title="Prediksi Arah Harga Saham",
    layout="centered"
)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model"]["path"]
THRESHOLD = config["model"]["threshold"]

# LOAD MODEL
model = joblib.load(MODEL_PATH)

# FEATURE ENGINEERING
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_20d"] = df["Close"].pct_change(20)

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    df["Volume_MA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]

    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()
    df["RSI_14"] = calculate_rsi(df["Close"])

    return df

# UI
st.title("📈 Prediksi Arah Harga Saham")
st.caption("Model MLOps – Multi-Stock Price Direction Prediction")

ticker = st.text_input(
    "Masukkan ticker saham (contoh: ASII.JK)",
    value="BBRI.JK"
)

if st.button("🔮 Prediksi"):
    try:
        df = yf.Ticker(ticker).history(period="3mo")

        if df.empty:
            st.error("❌ Data saham tidak ditemukan.")
            st.stop()

        # GRAFIK HARGA
        st.subheader("📊 Grafik Harga Penutupan")
        st.line_chart(df["Close"])

        # FEATURE ENGINEERING
        df_feat = prepare_features(df).dropna()

        for col in model.feature_names_in_:
            if col not in df_feat.columns:
                df_feat[col] = 0

        X = df_feat[model.feature_names_in_].iloc[[-1]]
        prob = model.predict_proba(X)[0][1]

        prediction = "NAIK 📈" if prob >= THRESHOLD else "TURUN 📉"

        # OUTPUT
        st.success(f"Prediksi: **{prediction}**")
        st.metric("Probabilitas Naik", f"{prob*100:.2f}%")

        # TIMESTAMP WIB
        wib = pytz.timezone("Asia/Jakarta")
        now_wib = datetime.now(wib)

        st.caption(f"⏱️ {now_wib.strftime('%Y-%m-%d %H:%M:%S')} WIB")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
