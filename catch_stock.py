# -*- coding: utf-8 -*-
"""
個股資料抓取 + 技術指標計算 + Firestore 更新與寫回
Yahoo Finance 穩定修正版（2025）
"""

import os
import json
import time
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ================== 參數 ==================
WRITE_DAYS = 3
COLLECTION = "NEW_stock_data_liteon"
PERIOD = "12mo"

# ================= Firebase 初始化 =================
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None

if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    print("⚠️ FIREBASE 未設定，Firestore 寫入將略過")

# ================= Yahoo 安全下載 =================
def safe_download(ticker, period="12mo", retry=3, sleep_sec=2):
    for i in range(retry):
        try:
            print(f"⬇️ downloading {ticker}")
            df = yf.download(
                tickers=ticker,
                period=period,
                interval="1d",
                progress=False,
                threads=False
            )
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            print(f"⚠️ {ticker} error {i+1}/{retry}: {e}")
        time.sleep(sleep_sec)

    print(f"❌ {ticker} 無法取得資料")
    return None

# ================= 技術指標 =================
def add_all_indicators(df):
    df = df.copy()

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(20).mean() / loss.rolling(20).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    denom = high14 - low14
    df["K"] = np.where(denom == 0, 50, 100 * (df["Close"] - low14) / denom)
    df["D"] = df["K"].rolling(3).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    return df.dropna()

# ================= 個股流程 =================
def fetch_prepare_recalc(ticker):
    df = safe_download(ticker, PERIOD)
    if df is None or len(df) == 0:
        return None
    return add_all_indicators(df)

def save_stock_recent_days(df, ticker):
    if db is None or df is None or len(df) == 0:
        return

    batch = db.batch()
    for idx, row in df.tail(WRITE_DAYS).iterrows():
        doc_ref = db.collection(COLLECTION).document(idx.strftime("%Y-%m-%d"))
        batch.set(doc_ref, {
            ticker: {
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Volume": float(row["Volume"]),
                "MACD": float(row["MACD"]),
                "RSI": float(row["RSI"]),
                "K": float(row["K"]),
                "D": float(row["D"]),
                "ATR_14": float(row["ATR_14"]),
            }
        }, merge=True)

    batch.commit()
    print(f"🔥 {ticker} 寫入完成")

# ================= 指數 / 外生因子 =================
def save_factor_latest(tickers, alias):
    if db is None:
        return

    for tk in tickers:
        df = safe_download(tk, PERIOD)
        if df is None or len(df) == 0:
            continue

        row = df.iloc[-1]
        date_str = df.index[-1].strftime("%Y-%m-%d")
        db.collection(COLLECTION).document(date_str).set({
            alias: {"Close": float(row["Close"])}
        }, merge=True)

        print(f"🔥 {alias} 更新成功（來源 {tk}）")
        return

    print(f"⚠️ {alias} 全部來源失敗")

# ================= Main =================
if __name__ == "__main__":

    for ticker in ["2301.TW", "2408.TW", "8110.TW"]:
        df = fetch_prepare_recalc(ticker)
        save_stock_recent_days(df, ticker)

    save_factor_latest(["^TWII"], "TAIEX")
    save_factor_latest(["^TELI", "IR0027.TW"], "ELECTRONICS")
    save_factor_latest(["^SOX", "SOXX", "SMH"], "SOX")
    save_factor_latest(["MU", "MU.VI", "MU.MX"], "MU_US")
    save_factor_latest(["TWD=X", "USDTWD=X"], "USD_TWD")

    print("✅ 全部完成")
