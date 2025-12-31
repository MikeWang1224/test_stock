# -*- coding: utf-8 -*-
"""
3481.TW | 6M Outlook
Quantile Attention LSTM (P10 / P50 / P90)
[FIXED: Proper Close scaling & anchor]
"""

# ===============================
# Imports
# ===============================
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from zoneinfo import ZoneInfo
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Multiply, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# ===============================
# Firebase
# ===============================
import firebase_admin
from firebase_admin import credentials, firestore

key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
if not key_dict:
    raise RuntimeError("❌ FIREBASE env missing")

try:
    firebase_admin.get_app()
except Exception:
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ===============================
# Config
# ===============================
TICKER = "3481.TW"
COLLECTION = "NEW_stock_data_liteon"

LOOKBACK = 40
FORECAST_DAYS = 120
EPOCHS = 80
BATCH = 32
LR = 5e-4

FEATURES = ["Close", "RSI", "MACD", "Volume"]
QUANTILES = [0.1, 0.5, 0.9]

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===============================
# Data Utils
# ===============================
def load_df(ticker):
    rows = []
    for doc in db.collection(COLLECTION).stream():
        p = doc.to_dict().get(ticker)
        if p:
            rows.append({"date": doc.id, **p})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")

def ensure_latest_row(df):
    today = pd.Timestamp(datetime.now().date())
    last = df.index.max()
    if last >= today:
        return df
    for d in pd.bdate_range(last, today)[1:]:
        df.loc[d] = df.loc[last]
    return df.sort_index()

def indicators(df):
    df = df.copy()
    delta = df["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - 100 / (1 + rs)

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

    return df.dropna()

# ===============================
# Dataset (FIXED)
# ===============================
def make_dataset(df):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_all = x_scaler.fit_transform(df[FEATURES])
    y_close = y_scaler.fit_transform(df[["Close"]])

    X, y = [], []
    for i in range(len(df) - LOOKBACK - FORECAST_DAYS):
        X.append(X_all[i:i+LOOKBACK])
        y.append(
            y_close[i+LOOKBACK:i+LOOKBACK+FORECAST_DAYS, 0]
        )

    return np.array(X), np.array(y), x_scaler, y_scaler

# ===============================
# Quantile Loss
# ===============================
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e))
    return loss

# ===============================
# Attention Block
# ===============================
def attention(x):
    score = Dense(1, activation="tanh")(x)
    score = Lambda(lambda z: K.squeeze(z, -1))(score)
    weights = Lambda(lambda z: K.softmax(z))(score)
    weights = Lambda(lambda z: K.expand_dims(z, -1))(weights)
    ctx = Multiply()([x, weights])
    return Lambda(lambda z: K.sum(z, axis=1))(ctx)

# ===============================
# Model
# ===============================
def build_model():
    inp = Input(shape=(LOOKBACK, len(FEATURES)))

    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=True)(x)

    ctx = attention(x)
    outs = [Dense(FORECAST_DAYS)(ctx) for _ in QUANTILES]

    model = Model(inp, outs)
    model.compile(
        optimizer=Adam(LR),
        loss=[quantile_loss(q) for q in QUANTILES]
    )
    return model

# ===============================
# Train
# ===============================
df = indicators(ensure_latest_row(load_df(TICKER)))
X, y, x_scaler, y_scaler = make_dataset(df)

split = int(len(X) * 0.8)
model = build_model()
model.fit(
    X[:split], [y[:split]]*3,
    validation_data=(X[split:], [y[split:]]*3),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[EarlyStopping(patience=12, restore_best_weights=True)],
    verbose=1
)

# ===============================
# Forecast (FIXED)
# ===============================
last_window = x_scaler.transform(df[FEATURES].iloc[-LOOKBACK:])
preds = model.predict(last_window[np.newaxis, ...], verbose=0)

p10, p50, p90 = [
    y_scaler.inverse_transform(p[0].reshape(-1,1)).flatten()
    for p in preds
]

# 🚑 quantile crossing guard
p10 = np.minimum(p10, p50)
p90 = np.maximum(p90, p50)

future_dates = pd.bdate_range(
    start=df.index[-1] + BDay(1),
    periods=FORECAST_DAYS
)

# ===============================
# Plot
# ===============================
STEP = 20
idx = np.arange(0, FORECAST_DAYS, STEP)

dates = future_dates[idx]
mid, low, high = p50[idx], p10[idx], p90[idx]

today_price = df["Close"].iloc[-1]

fig, ax = plt.subplots(figsize=(14, 8))
ax.fill_between(dates, low, high, alpha=0.18, label="Expected Range (10–90%)")
ax.plot(dates, mid, color="red", lw=3, marker="o", label="Projected Path")

ax.scatter(
    df.index[-1],
    today_price,
    s=240,
    marker="*",
    color="orange",
    edgecolor="black",
    label="Today",
    zorder=6
)

offset = (high.max() - low.min()) * 0.03
for d, p in zip(dates, mid):
    ax.text(d, p + offset, f"{p:.2f}", ha="center", fontsize=11)

ax.set_title("3481.TW · 6M Outlook (Quantile Attention LSTM)", fontsize=15)
ax.grid(alpha=0.3)
ax.legend()

now_tw = datetime.now(ZoneInfo("Asia/Taipei"))
ax.text(
    0.01, 0.01,
    f"Generated: {now_tw:%Y-%m-%d %H:%M} (UTC+8)",
    transform=ax.transAxes,
    fontsize=9,
    alpha=0.65
)

out = os.path.join(
    RESULT_DIR,
    f"{datetime.now():%Y-%m-%d}_3481_quantile_outlook_fixed.png"
)
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()

print(f"Saved → {out}")
