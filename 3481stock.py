# -*- coding: utf-8 -*-
"""
FireBase_Attention_LSTM_3481_6M_Huadong.py

群創 3481.TW
- Attention-LSTM
- 六個月趨勢預測圖（華東方法）
- 不畫 5D、不回測
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.offsets import BDay
from zoneinfo import ZoneInfo

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Softmax, Lambda
from tensorflow.keras.callbacks import EarlyStopping

# ================= 基本設定 =================
TICKER = "3481.TW"
COLLECTION = "NEW_stock_data_liteon"

LOOKBACK = 60
STEPS = 5
TREND_H = 20
K_FLAT = 1.1
FORECAST_6M_DAYS = 120

FEATURES = [
    "Close", "Open", "High", "Low",
    "Volume", "RSI", "MACD", "K", "D", "ATR_14",
    "VOL_REL", "TREND_60", "TREND_SLOPE_20"
]

now_tw = datetime.now(ZoneInfo("Asia/Taipei"))
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ================= Firebase =================
import firebase_admin
from firebase_admin import credentials, firestore

key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
cred = credentials.Certificate(key_dict)
try:
    firebase_admin.get_app()
except Exception:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ================= Firestore 讀取 =================
def load_df_from_firestore(ticker, collection, days=700):
    rows = []
    for doc in db.collection(collection).stream():
        p = doc.to_dict().get(ticker)
        if p:
            rows.append({"date": doc.id, **p})

    if not rows:
        raise ValueError("Firestore 無資料")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days).set_index("date")
    return df

def ensure_latest_trading_row(df):
    today = pd.Timestamp(datetime.now().date())
    last = df.index.max()
    if last.normalize() >= today:
        return df
    for d in pd.bdate_range(last, today)[1:]:
        df.loc[d] = df.loc[last]
    return df.sort_index()

# ================= Feature =================
def add_features(df):
    df = df.copy()
    df["Volume"] = np.log1p(df["Volume"].astype(float))

    close = df["Close"].astype(float)
    logret = np.log(close).diff()

    df["RET_STD_20"] = logret.rolling(20).std()
    df["VOL_REL"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

    ma60 = close.rolling(60)
    df["TREND_60"] = (close - ma60.mean()) / (ma60.std() + 1e-9)
    df["TREND_SLOPE_20"] = close.rolling(20).mean().diff() / close

    return df

# ================= Sequence =================
def create_sequences(df, features):
    X, y_ret, y_dir, y_tr3 = [], [], [], []

    close = df["Close"].astype(float)
    logret = np.log(close).diff()
    feat = df[features].values

    for i in range(LOOKBACK, len(df) - max(STEPS, TREND_H)):
        x = feat[i-LOOKBACK:i]
        if np.any(np.isnan(x)):
            continue

        scale = df["RET_STD_20"].iloc[i-1]
        if not np.isfinite(scale) or scale < 1e-9:
            continue

        r5 = logret.iloc[i:i+STEPS].values
        if np.any(np.isnan(r5)):
            continue

        y_ret.append(r5 / scale)
        y_dir.append(1.0 if r5.sum() > 0 else 0.0)

        r_tr = logret.iloc[i:i+TREND_H].values
        cum = r_tr.sum()
        thr = K_FLAT * scale * np.sqrt(TREND_H)

        if cum > thr:
            cls = 2
        elif cum < -thr:
            cls = 0
        else:
            cls = 1

        oh = np.zeros(3)
        oh[cls] = 1.0

        X.append(x)
        y_tr3.append(oh)

    return np.array(X), np.array(y_ret), np.array(y_dir), np.array(y_tr3)

# ================= Model =================
def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.25)(x)

    score = Dense(1)(x)
    w = Softmax(axis=1)(score)
    ctx = Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1))([x, w])

    ret = Dense(STEPS, activation="tanh")(ctx)
    ret = Lambda(lambda t: t * 2.0, name="return")(ret)

    direction = Dense(1, activation="sigmoid", name="direction")(ctx)
    trend3 = Dense(3, activation="softmax", name="trend3")(ctx)

    model = Model(inp, [ret, direction, trend3])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(4e-4),
        loss={
            "return": tf.keras.losses.Huber(),
            "direction": "binary_crossentropy",
            "trend3": "categorical_crossentropy"
        },
        loss_weights={
            "return": 1.0,
            "direction": 0.3,
            "trend3": 1.3
        }
    )
    return model

# ================= 6M Plot（華東方法） =================
def plot_6m_forecast_huadong(df, trend_prob):
    last_date = df.index[-1]
    last_close = float(df["Close"].iloc[-1])

    logret = np.log(df["Close"]).diff()
    sigma = float(logret.rolling(20).std().iloc[-1])
    sigma = np.clip(sigma, 0.005, 0.02)

    p_down, p_flat, p_up = trend_prob  # softmax 機率
    drift = (p_up - p_down) * sigma * 0.35
    drift = np.clip(drift, -0.0015, 0.0015)

    future_days = pd.bdate_range(last_date + BDay(1), periods=FORECAST_6M_DAYS)
    t = np.arange(1, FORECAST_6M_DAYS + 1)

    mean_path = last_close * np.exp(drift * t)
    band = sigma * np.sqrt(t)

    upper = mean_path * np.exp(+1.2 * band)
    lower = mean_path * np.exp(-1.2 * band)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-120:], df["Close"].iloc[-120:], color="black", label="History")
    plt.plot(future_days, mean_path, linestyle="--", label="6M Mean Trend")
    plt.fill_between(future_days, lower, upper, alpha=0.25, label="Reasonable Band")

    plt.title(
        f"{TICKER} | 6M Outlook (Huadong Method)\n"
        f"Up {p_up:.0%} | Flat {p_flat:.0%} | Down {p_down:.0%}"
    )
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)

    fname = f"results/{datetime.now().strftime('%Y-%m-%d')}_{TICKER}_6M_forecast_huadong.png"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"6M forecast saved (Huadong): {fname}")

# ================= Main =================
if __name__ == "__main__":

    df = load_df_from_firestore(TICKER, COLLECTION)
    df = ensure_latest_trading_row(df)
    df = add_features(df)
    df = df.dropna()

    X, y_ret, y_dir, y_tr3 = create_sequences(df, FEATURES)

    split = int(len(X) * 0.85)
    X_tr, X_va = X[:split], X[split:]
    y_ret_tr, y_ret_va = y_ret[:split], y_ret[split:]
    y_dir_tr, y_dir_va = y_dir[:split], y_dir[split:]
    y_tr3_tr, y_tr3_va = y_tr3[:split], y_tr3[split:]

    scaler = MinMaxScaler()
    scaler.fit(df[FEATURES].values)

    def scale(Xb):
        n, t, f = Xb.shape
        return scaler.transform(Xb.reshape(-1, f)).reshape(n, t, f)

    X_tr_s = scale(X_tr)
    X_va_s = scale(X_va)

    model = build_model((LOOKBACK, len(FEATURES)))
    model.fit(
        X_tr_s,
        {"return": y_ret_tr, "direction": y_dir_tr, "trend3": y_tr3_tr},
        validation_data=(X_va_s, {"return": y_ret_va, "direction": y_dir_va, "trend3": y_tr3_va}),
        epochs=120,
        batch_size=16,
        callbacks=[EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)],
        verbose=2
    )

    model.save(f"models/{TICKER}_attn_lstm_6M_Huadong.keras")

    _, _, pred_tr3 = model.predict(X_va_s, verbose=0)
    plot_6m_forecast_huadong(df, pred_tr3[-1])
