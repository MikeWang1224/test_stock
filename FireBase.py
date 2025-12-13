# -*- coding: utf-8 -*-
"""
FireBase.py
- Firestore 讀 OHLCV
- 技術指標即時計算
- 假日補今天
- LSTM multi-step
- 畫圖輸出 results
- 不上傳 Storage（避免付費問題）
"""

import os, math, json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

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

# ================= Firestore 讀取 =================
def load_df_from_firestore(ticker, collection="NEW_stock_data_liteon", days=400):
    rows = []
    if db:
        for doc in db.collection(collection).stream():
            p = doc.to_dict().get(ticker)
            if p:
                rows.append({"date": doc.id, **p})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("⚠️ Firestore 無資料")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days).set_index("date")
    return df

# ================= 假日補今天 =================
def ensure_today_row(df):
    today = pd.Timestamp(datetime.now().date())
    last_date = df.index.max()
    if last_date < today:
        df.loc[today] = df.loc[last_date]
        print(f"⚠️ 今日無資料，使用 {last_date.date()} 補今日")
    return df.sort_index()

# ================= 技術指標 =================
def add_features(df):
    df = df.copy()

    df["RET_1"] = df["Close"].pct_change()
    df["LOG_RET_1"] = np.log(df["Close"] / df["Close"].shift())

    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["Close_minus_SMA5"] = df["Close"] - df["SMA5"]
    df["SMA5_minus_SMA10"] = df["SMA5"] - df["SMA10"]

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_width"] = (2 * std20) / ma20

    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (direction * df["Volume"]).cumsum()
    df["OBV_SMA_20"] = df["OBV"].rolling(20).mean()
    df["Vol_SMA_5"] = df["Volume"].rolling(5).mean()

    return df.dropna()

# ================= LSTM helpers =================
def create_sequences(df, features, steps=10, window=60):
    X, y = [], []
    data = df[features].values
    closes = df["Close"].values
    for i in range(window, len(df) - steps + 1):
        X.append(data[i-window:i])
        y.append(closes[i:i+steps])
    return np.array(X), np.array(y)

def build_lstm(input_shape, steps):
    m = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(steps)
    ])
    m.compile(optimizer="adam", loss="mae")
    return m

# ================= 畫圖 =================
def plot_and_save(df_hist, future_df):
    hist = df_hist.tail(10)
    if hist.empty:
        hist = df_hist.tail(1)  # 至少取一筆

    plt.figure(figsize=(16,8))
    plt.plot(hist.index, hist["Close"], label="Close")
    plt.plot(hist.index, hist["SMA5"], label="SMA5")
    plt.plot(hist.index, hist["SMA10"], label="SMA10")

    # 接上預測
    start_date = hist.index[-1]
    plt.plot(
        [start_date] + list(future_df["date"]),
        [hist["Close"].iloc[-1]] + list(future_df["Pred_Close"]),
        "r:o", label="Pred Close"
    )

    plt.plot(future_df["date"], future_df["Pred_MA5"], "--", label="Pred MA5")
    plt.plot(future_df["date"], future_df["Pred_MA10"], "--", label="Pred MA10")

    plt.legend()
    plt.title("2301.TW LSTM 預測")

    os.makedirs("results", exist_ok=True)
    fname = f"{datetime.now().strftime('%Y-%m-%d')}_pred.png"
    fpath = os.path.join("results", fname)
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 圖片已輸出：{fpath}")
    return fpath

# ================= Main =================
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 60
    STEPS = 10

    df = load_df_from_firestore(TICKER)
    df = ensure_today_row(df)
    df = add_features(df)

    FEATURES = [
        "Close", "Volume",
        "RET_1", "LOG_RET_1",
        "Close_minus_SMA5", "SMA5_minus_SMA10",
        "ATR_14", "BB_width",
        "OBV", "OBV_SMA_20",
        "Vol_SMA_5"
    ]

    X, y = create_sequences(df, FEATURES, STEPS, LOOKBACK)
    split = int(len(X) * 0.85)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    sx = MinMaxScaler()
    X_tr_s = sx.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    X_te_s = sx.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)

    sy = MinMaxScaler()
    y_tr_s = sy.fit_transform(y_tr)

    model = build_lstm((LOOKBACK, len(FEATURES)), STEPS)
    model.fit(X_tr_s, y_tr_s, epochs=50, batch_size=32, verbose=2,
              callbacks=[EarlyStopping(patience=6, restore_best_weights=True)])

    preds = sy.inverse_transform(model.predict(X_te_s))
    last_window_closes = X_te[-1][:,0]

    # 預測 MA
    seq = list(last_window_closes)
    future = []
    for p in preds[-1]:
        seq.append(p)
        future.append({
            "Pred_Close": p,
            "Pred_MA5": np.mean(seq[-5:]),
            "Pred_MA10": np.mean(seq[-10:])
        })

    start = (pd.Timestamp(datetime.now().date()) + BDay(1))
    future_df = pd.DataFrame(future)
    future_df["date"] = pd.bdate_range(start=start, periods=STEPS)

    plot_and_save(df, future_df)
