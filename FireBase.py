# -*- coding: utf-8 -*-
"""
FireBase_LSTM_v2.py
- Firestore 讀 OHLCV + 已算好技術指標
- 不重算指標（避免分佈錯亂）
- 預測 log return（多步）
- 價格由 return 還原
- 原預測圖不動
- 新增：預測回測誤差圖（Pred vs Actual）
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
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

# ================= Sequence（預測 log return） =================
def create_sequences(df, features, steps=10, window=60):
    X, y = [], []

    logret = np.log(df["Close"]).diff()
    df = df.iloc[1:]          # 丟掉第一筆，避免 NaN
    logret = logret.iloc[1:]  # 與 df 對齊
    data = df[features].values

    for i in range(window, len(df) - steps):
        X.append(data[i - window:i])
        y.append(logret.iloc[i:i + steps].values)

    return np.array(X), np.array(y)

# ================= LSTM =================
def build_lstm(input_shape, steps):
    m = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.1),
        Dense(steps)
    ])
    m.compile(optimizer="adam", loss="huber")
    return m

# ================= 原預測圖（完全不動） =================
def plot_and_save(df_hist, future_df):
    hist = df_hist.tail(10)

    hist_dates = hist.index.strftime("%Y-%m-%d").tolist()
    future_dates = future_df["date"].dt.strftime("%Y-%m-%d").tolist()

    all_dates = hist_dates + future_dates
    x_hist = np.arange(len(hist_dates))
    x_future = np.arange(len(hist_dates), len(all_dates))

    plt.figure(figsize=(18,8))
    ax = plt.gca()

    ax.plot(x_hist, hist["Close"], label="Close")
    ax.plot(x_hist, hist["SMA5"], label="SMA5")
    ax.plot(x_hist, hist["SMA10"], label="SMA10")

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["Close"].iloc[-1]] + future_df["Pred_Close"].tolist(),
        "r:o", label="Pred Close"
    )

    for i, price in enumerate(future_df["Pred_Close"]):
        ax.text(x_future[i], price + 0.3, f"{price:.2f}",
                color="red", fontsize=9, ha="center")

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["SMA5"].iloc[-1]] + future_df["Pred_MA5"].tolist(),
        "g--o", label="Pred MA5"
    )

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["SMA10"].iloc[-1]] + future_df["Pred_MA10"].tolist(),
        "b--o", label="Pred MA10"
    )

    ax.set_xticks(np.arange(len(all_dates)))
    ax.set_xticklabels(all_dates, rotation=45, ha="right")

    ax.legend()
    ax.set_title("2301.TW LSTM 預測（Return-based 穩定版）")

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{datetime.now():%Y-%m-%d}_pred.png",
                dpi=300, bbox_inches="tight")
    plt.close()

# ================= 新增：回測誤差圖 =================
def plot_backtest_error(df, X_te_s, y_te, model, steps):
    """
    使用測試集最後一筆，畫 Pred vs Actual（x 軸 = 交易日）
    """
    X_last = X_te_s[-1:]
    y_true = y_te[-1]

    pred_ret = model.predict(X_last)[0]

    # 對應的實際交易日（最後 steps 天）
    dates = df.index[-steps:]

    # 對應起始價格（回測起點前一天）
    start_price = df.loc[dates[0] - BDay(1), "Close"]

    true_prices = []
    pred_prices = []

    p_true = start_price
    p_pred = start_price

    for r_t, r_p in zip(y_true, pred_ret):
        p_true *= np.exp(r_t)
        p_pred *= np.exp(r_p)
        true_prices.append(p_true)
        pred_prices.append(p_pred)

    mae = np.mean(np.abs(np.array(true_prices) - np.array(pred_prices)))
    rmse = np.sqrt(np.mean((np.array(true_prices) - np.array(pred_prices)) ** 2))

    plt.figure(figsize=(12,6))
    plt.plot(dates, true_prices, label="Actual Close")
    plt.plot(dates, pred_prices, "--o", label="Pred Close")
    plt.title(f"Backtest Prediction | MAE={mae:.2f}, RMSE={rmse:.2f}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig(
        f"results/{datetime.now():%Y-%m-%d}_backtest.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ================= Main =================
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 60
    STEPS = 10

    df = load_df_from_firestore(TICKER)
    df = ensure_today_row(df)

    FEATURES = [
        "Close",
        "Volume",
        "RSI",
        "MACD",
        "K",
        "D",
        "ATR_14"
    ]

    # SMA 只為畫圖
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df = df.dropna()

    X, y = create_sequences(df, FEATURES, STEPS, LOOKBACK)
    split = int(len(X) * 0.85)

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    sx = MinMaxScaler()
    sx.fit(df[FEATURES].iloc[:split + LOOKBACK])

    def scale_X(X):
        n, t, f = X.shape
        return sx.transform(X.reshape(-1, f)).reshape(n, t, f)

    X_tr_s = scale_X(X_tr)
    X_te_s = scale_X(X_te)

    model = build_lstm((LOOKBACK, len(FEATURES)), STEPS)
    model.fit(
        X_tr_s, y_tr,
        epochs=50,
        batch_size=32,
        verbose=2,
        callbacks=[EarlyStopping(patience=6, restore_best_weights=True)]
    )

    # ===== 預測未來 =====
    raw_returns = model.predict(X_te_s)[-1]

    today = pd.Timestamp(datetime.now().date())
    last_trade_date = df.index[df.index < today][-1]
    last_close = df.loc[last_trade_date, "Close"]

    prices = []
    price = last_close
    for r in raw_returns:
        price *= np.exp(r)
        prices.append(price)

    seq = df.loc[:last_trade_date, "Close"].iloc[-10:].tolist()
    future = []

    for p in prices:
        seq.append(p)
        future.append({
            "Pred_Close": p,
            "Pred_MA5": np.mean(seq[-5:]),
            "Pred_MA10": np.mean(seq[-10:])
        })

    future_df = pd.DataFrame(future)
    future_df["date"] = pd.bdate_range(
        start=df.index.max() + BDay(1),
        periods=STEPS
    )

    plot_and_save(df, future_df)
    plot_backtest_error(df, X_te_s, y_te, model, STEPS)
