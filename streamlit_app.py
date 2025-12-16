import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt

# LOAD CONFIG & MODEL
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model"]["path"]
THRESHOLD = config["model"]["threshold"]
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

# STREAMLIT UI
st.set_page_config(page_title="Stock Direction Prediction", layout="wide")

st.title("📈 Prediksi Arah Harga Saham")
st.write("Aplikasi MLOps untuk memprediksi **arah harga saham (UP / DOWN)**")

ticker = st.text_input("Masukkan Kode Saham (contoh: BBCA.JK)", "BBCA.JK")

if st.button("🔍 Prediksi"):
    df = yf.Ticker(ticker).history(period="6mo")

    if df.empty:
        st.error("Data saham tidak ditemukan!")
    else:
        df = prepare_features(df)
        df.dropna(inplace=True)

        # PREDICTION
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0

        X = df[model.feature_names_in_].iloc[[-1]]
        prob = model.predict_proba(X)[0][1]

        prediction = "📈 UP" if prob >= THRESHOLD else "📉 DOWN"

        st.subheader("🧠 Hasil Prediksi")
        st.metric("Prediksi Arah", prediction)
        st.metric("Probabilitas Naik", f"{prob:.2%}")

        # PRICE CHART
        st.subheader("📊 Grafik Harga Saham")

        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close Price")
        ax.plot(df.index, df["MA_20"], label="MA 20")
        ax.plot(df.index, df["MA_50"], label="MA 50")
        ax.legend()
        ax.set_title(f"Harga Saham {ticker}")
        st.pyplot(fig)

        # RSI CHART
        st.subheader("📉 RSI Indicator")

        fig2, ax2 = plt.subplots()
        ax2.plot(df.index, df["RSI_14"])
        ax2.axhline(70, linestyle="--")
        ax2.axhline(30, linestyle="--")
        ax2.set_title("RSI 14")
        st.pyplot(fig2)
