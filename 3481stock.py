# -*- coding: utf-8 -*-
"""
3481.TW 群創光電
6-Month Forecast (MULTI-STEP ATTENTION LSTM)
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Permute, Multiply, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# ===============================
# Firebase Setup
# ===============================
import firebase_admin
from firebase_admin import credentials, firestore

key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
if not key_dict:
    raise ValueError("❌ FIREBASE 環境變數未設定")

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
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

LOOKBACK = 40
FORECAST_DAYS = 120   # 6 months
EPOCHS = 80
BATCH = 32
LR = 5e-4

FEATURES = ["Close", "RSI", "MACD", "Volume"]

# ===============================
# Utils
# ===============================
def load_df_from_firestore(ticker, collection=COLLECTION, days=600):
    rows = []
    for doc in db.collection(collection).stream():
        p = doc.to_dict().get(ticker)
        if p:
            rows.append({"date": doc.id, **p})

    if not rows:
        raise ValueError(f"⚠️ Firestore 無 {ticker} 資料")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days).set_index("date")
    return df

def ensure_latest_trading_row(df):
    today = pd.Timestamp(datetime.now().date())
    last = df.index.max()
    if last >= today:
        return df

    for d in pd.bdate_range(last, today)[1:]:
        df.loc[d] = df.loc[last]

    return df.sort_index()

def compute_indicators(df):
    df = df.copy()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

    return df.dropna()

def make_dataset(df):
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(df[FEATURES])

    close_idx = FEATURES.index("Close")

    X, y = [], []
    for i in range(len(X_all) - LOOKBACK - FORECAST_DAYS):
        X.append(X_all[i:i+LOOKBACK])
        y.append(
            X_all[
                i+LOOKBACK : i+LOOKBACK+FORECAST_DAYS,
                close_idx
            ]
        )

    return np.array(X), np.array(y), scaler

# ===============================
# Attention Block
# ===============================
def attention_block(x):
    # x: (batch, time, features)
    score = Dense(1, activation="tanh")(x)
    score = Lambda(lambda z: K.squeeze(z, axis=-1))(score)
    weights = Lambda(lambda z: K.softmax(z))(score)
    weights = Lambda(lambda z: K.expand_dims(z, axis=-1))(weights)
    context = Multiply()([x, weights])
    context = Lambda(lambda z: K.sum(z, axis=1))(context)
    return context

def build_model():
    inp = Input(shape=(LOOKBACK, len(FEATURES)))

    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=True)(x)

    context = attention_block(x)
    out = Dense(FORECAST_DAYS)(context)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(LR),
        loss="mse"
    )
    return model

def add_timestamp(ax):
    now_tw = datetime.now(ZoneInfo("Asia/Taipei"))
    ax.text(
        0.01, 0.01,
        f"Generated: {now_tw:%Y-%m-%d %H:%M} (UTC+8)",
        transform=ax.transAxes,
        fontsize=9,
        color="gray",
        alpha=0.65,
        ha="left",
        va="bottom"
    )

# ===============================
# Main
# ===============================
print("Loading Firebase data...")
df = load_df_from_firestore(TICKER)
df = ensure_latest_trading_row(df)
df = compute_indicators(df)

X, y, scaler = make_dataset(df)

split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print("Training Attention-LSTM (Multi-step)...")
model = build_model()
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[
        EarlyStopping(
            patience=12,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# ===============================
# Forecast
# ===============================
print("Running 6-month forecast...")

last_window = df[FEATURES].iloc[-LOOKBACK:]
scaled_last = scaler.transform(last_window)

pred_scaled = model.predict(
    scaled_last[np.newaxis, ...],
    verbose=0
)[0]

close_idx = FEATURES.index("Close")
close_min = scaler.data_min_[close_idx]
close_max = scaler.data_max_[close_idx]

future_close = (
    pred_scaled * (close_max - close_min) + close_min
)

future_dates = pd.bdate_range(
    start=df.index[-1] + BDay(1),
    periods=FORECAST_DAYS
)

# ===============================
# Plot
# ===============================
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(
    df.index[-120:],
    df["Close"].iloc[-120:],
    label="Historical",
    color="black"
)
ax.plot(
    future_dates,
    future_close,
    label="6M Forecast (Multi-step)",
    color="tab:red"
)

ax.set_title(
    "3481.TW | 6-Month Forecast (Attention-LSTM, Multi-step)",
    fontsize=14
)
ax.legend()
ax.grid(alpha=0.3)
add_timestamp(ax)

out_path = os.path.join(
    RESULT_DIR,
    f"{datetime.now():%Y-%m-%d}_3481_6m_forecast_multistep.png"
)
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Saved → {out_path}")
