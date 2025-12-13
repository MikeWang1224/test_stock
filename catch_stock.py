# -*- coding: utf-8 -*-
"""
個股資料抓取 + 技術指標計算 + Firestore 更新與寫回
不含模型、不含預測、不含繪圖
"""
 
import os, json
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


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

# ---------------- 傳統技術指標：SMA / RSI / KD / MACD ----------------
def add_basic_indicators(df):
    df = df.copy()

    # --- SMA ---
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # --- RSI (20) ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=20).mean()
    avg_loss = loss.rolling(window=20).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- KD (14,3) ---
    df['Lowest_14'] = df['Low'].rolling(window=14).min()
    df['Highest_14'] = df['High'].rolling(window=14).max()
    denom = (df['Highest_14'] - df['Lowest_14'])
    df['K'] = np.where(denom == 0, 50.0, 100 * (df['Close'] - df['Lowest_14']) / denom)
    df['D'] = df['K'].rolling(window=3).mean()

    # --- MACD (12,26,9) ---
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['SignalLine'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

# ---------------- 其他技術特徵 ----------------
def add_technical_features(df):
    df = df.copy()

    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()

    df['RET_1'] = df['Close'].pct_change().fillna(0)
    df['LOG_RET_1'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)

    df['Close_minus_SMA5'] = df['Close'] - df['SMA_5']
    df['SMA5_minus_SMA10'] = df['SMA_5'] - df['SMA_10']

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i-1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i-1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_SMA_20'] = df['OBV'].rolling(20).mean()

    df['Vol_SMA_5'] = df['Volume'].rolling(5).mean()
    df['Vol_SMA_20'] = df['Volume'].rolling(20).mean()

    df = df.dropna()
    return df

# ---------------- 抓資料 + 加指標 ----------------
def fetch_and_prepare(ticker="2301.TW", period="12mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    df = add_technical_features(df)
    df = add_basic_indicators(df)
    df = df.dropna()
    return df

# ---------------- Firestore 更新今日 Close（如果存在） ----------------
def update_today_from_firestore(df, ticker="2301.TW"):
    if db is None:
        return df

    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        doc = db.collection("NEW_stock_data_liteon").document(today_str).get()
        if doc.exists:
            payload = doc.to_dict().get(ticker, {})
            if "Close" in payload:
                df.loc[pd.Timestamp(today_str), "Close"] = float(payload["Close"])
                print(f"已從 Firestore 更新今日 Close = {payload['Close']}")
    except:
        pass

    return df.dropna()

# ---------------- Firestore 寫回歷史資料 ----------------
def save_stock_data_to_firestore(df, ticker="2301.TW", collection="NEW_stock_data_liteon"):
    if db is None:
        print("⚠️ FIREBASE 未啟用，略過寫入")
        return

    batch = db.batch()
    count = 0

    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        payload = {
            "Close": float(row["Close"]),
            "Volume": float(row["Volume"]),
            "MACD": float(row["MACD"]),
            "RSI": float(row["RSI"]),
            "K": float(row["K"]),
            "D": float(row["D"])
        }

        doc_ref = db.collection(collection).document(date_str)
        batch.set(doc_ref, {ticker: payload})

        count += 1
        if count >= 300:
            batch.commit()
            batch = db.batch()
            count = 0

    if count > 0:
        batch.commit()

    print(f"🔥 已寫入 Firestore：{collection}")

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    TICKER = "2301.TW"

    df = fetch_and_prepare(TICKER)
    df = update_today_from_firestore(df, TICKER)
    save_stock_data_to_firestore(df, TICKER)

    print(df.tail())
