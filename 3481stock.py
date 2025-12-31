# -*- coding: utf-8 -*-
"""
3481.TW | 6M Outlook
Quantile Attention LSTM (LOG-RETURN → REAL PRICE)
"""

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
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Lambda
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
    firebase_admin.initialize_app(credentials.Certificate(key_dict))

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

FEATURES = ["RET", "RSI", "MACD", "Volume"]
QUANTILES = [0.1, 0.5, 0.9]

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===============================
# Data
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

    df["RET"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()

# ===============================
# Dataset
# ===============================
def make_dataset(df):
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(df[FEATURES])

    ret_idx = FEATURES.index("RET")
    X, y = [], []

    for i in range(len(X_all) - LOOKBACK - FORECAST_DAYS):
        X.append(X_all[i:i+LOOKBACK])
        y.append(X_all[i+LOOKBACK:i+LOOKBACK+FORECAST_DAYS, ret_idx])

    return np.array(X), np.array(y), scaler

# ===============================
# Quantile Loss
# ===============================
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e))
    return loss

# ===============================
# Attention
# ===============================
def attention(x):
    s = Dense(1, activation="tanh")(x)
    s = Lambda(lambda z: K.squeeze(z, -1))(s)
    w = Lambda(lambda z: K.softmax(z))(s)
    w = Lambda(lambda z: K.expand_dims(z, -1))(w)
    return Lambda(lambda z: K.sum(z[0] * z[1], axis=1))([x, w])

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
df = indicators(load_df(TICKER))
X, y, scaler = make_dataset(df)

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
# Forecast (Return → Real Price)
# ===============================
last_price = df["Close"].iloc[-1]

last_window = scaler.transform(df[FEATURES].iloc[-LOOKBACK:])
preds = model.predict(last_window[np.newaxis], verbose=0)

ret_min, ret_max = scaler.data_min_[0], scaler.data_max_[0]
p10, p50, p90 = [(p[0] * (ret_max - ret_min) + ret_min) for p in preds]

# enforce monotonic
p10 = np.minimum(p10, p50)
p90 = np.maximum(p90, p50)

def to_price(ret):
    return last_price * np.exp(np.cumsum(ret))

p10_p, p50_p, p90_p = map(to_price, [p10, p50, p90])

future_dates = pd.bdate_range(df.index[-1] + BDay(1), periods=FORECAST_DAYS)

# ===============================
# Plot
# ===============================
STEP = 20
idx = np.arange(0, FORECAST_DAYS, STEP)

dates = future_dates[idx]
mid = p50_p[idx]
low = p10_p[idx]
high = p90_p[idx]

fig, ax = plt.subplots(figsize=(14, 8))
ax.fill_between(dates, low, high, alpha=0.18, label="Expected Range (10–90%)")
ax.plot(dates, mid, color="red", lw=3, marker="o", label="Projected Path")

ax.scatter(
    df.index[-1],
    last_price,
    s=220, marker="*",
    color="orange", edgecolor="black",
    label="Today", zorder=5
)

for d, p in zip(dates, mid):
    ax.text(d, p * 1.01, f"{p:.2f}", ha="center", fontsize=11)

ax.set_title("3481.TW · 6M Outlook (Quantile Attention LSTM)", fontsize=15)
ax.grid(alpha=0.3)
ax.legend()

now = datetime.now(ZoneInfo("Asia/Taipei"))
ax.text(0.01, 0.01, f"Generated: {now:%Y-%m-%d %H:%M} (UTC+8)",
        transform=ax.transAxes, fontsize=9, alpha=0.65)

out = f"{RESULT_DIR}/{now:%Y-%m-%d}_3481_quantile_REALPRICE.png"
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()

print(f"Saved → {out}")
