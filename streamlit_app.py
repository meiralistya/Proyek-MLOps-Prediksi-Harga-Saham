import streamlit as st
import joblib
import yaml
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Stock Direction Prediction", layout="centered")

# LOAD CONFIG
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model"]["path"]
THRESHOLD = config["model"]["threshold"]

model = joblib.load(MODEL_PATH)

st.title("📈 Prediksi Arah Harga Saham")
st.write("Prediksi **UP / DOWN** berdasarkan data historis 3 bulan")

ticker = st.text_input("Masukkan kode saham", value="ASII.JK")

@st.cache_data(ttl=3600)
def load_data(ticker):
    return yf.download(
        ticker,
        period="3mo",
        interval="1d",
        progress=False,
        threads=False
    )

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

if st.button("🔮 Prediksi"):
    try:
        df = load_data(ticker)
        if df.empty:
            st.error("Data saham tidak ditemukan")
            st.stop()

        df = prepare_features(df).dropna()

        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0

        X = df[model.feature_names_in_].iloc[[-1]]
        prob = model.predict_proba(X)[0][1]

        st.success(f"📌 Prediksi: **{'UP 📈' if prob >= THRESHOLD else 'DOWN 📉'}**")
        st.metric("Probabilitas Naik", f"{prob:.2%}")

    except Exception:
        st.error("⚠️ Yahoo Finance sedang membatasi request. Silakan coba lagi nanti.")
