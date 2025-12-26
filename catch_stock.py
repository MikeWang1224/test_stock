# -*- coding: utf-8 -*-

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from datetime import datetime
from finmind.data import DataLoader

# ---------------- FinMind 初始化 ----------------
FINMIND_TOKEN = os.getenv("FINMIND_API_TOKEN")
api = DataLoader()
api.login_by_token(FINMIND_TOKEN)

# ---------------- Firebase 初始化 ----------------
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

# ---------------- 技術指標計算 ----------------
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rs = gain.rolling(20).mean() / loss.rolling(20).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["K"] = 100 * (df["Close"] - low14) / (high14 - low14)
    df["D"] = df["K"].rolling(3).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["SignalLine"] = df["MACD"].ewm(span=9, adjust=False).mean()

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    return df.dropna()

# ---------------- Firestore 覆寫今日 Close ----------------
def overwrite_today_close(df, ticker, collection):
    if db is None:
        return df

    today = datetime.now().strftime("%Y-%m-%d")
    doc = db.collection(collection).document(today).get()
    if doc.exists:
        payload = doc.to_dict().get(ticker)
        if payload and "Close" in payload:
            ts = pd.Timestamp(today)
            if ts in df.index:
                df.loc[ts, "Close"] = payload["Close"]
    return df

# ---------------- FinMind 抓個股 ----------------
def fetch_stock(stock_id, period="12mo"):
    df = api.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=(pd.Timestamp.today() - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    )

    if df.empty:
        return df

    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume",
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]

# ---------------- Firestore 寫入 ----------------
def save_to_firestore(df, ticker, collection):
    if db is None or df.empty:
        return

    batch = db.batch()
    for idx, row in df.iterrows():
        doc_ref = db.collection(collection).document(idx.strftime("%Y-%m-%d"))
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
    print(f"🔥 Firestore 寫入完成：{ticker}")

# ---------------- Main ----------------
if __name__ == "__main__":
    COLLECTION = "NEW_stock_data_liteon"

    for stock in ["2301", "2408", "8110"]:
        df = fetch_stock(stock)
        df = overwrite_today_close(df, stock, COLLECTION)
        df = add_all_indicators(df)
        save_to_firestore(df, stock, COLLECTION)

    print("✅ FinMind 台股資料更新完成")
