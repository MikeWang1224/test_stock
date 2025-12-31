# -*- coding: utf-8 -*-
"""
3481.TW 群創光電
6-Month Forecast (RECURSIVE VERSION - FIXED)
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

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
FORECAST_DAYS = 120
EPOCHS = 60
BATCH = 32
LR = 5e-4

FEATURES = ["Close", "RSI", "MACD", "Volume"]

# ===============================
# Utils
# ===============================
def load_df_from_firestore(ticker, collection=COLLECTION, days=500):
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

    X, y = [], []
    close_idx = FEATURES.index("Close")

    for i in range(len(X_all) - LOOKBACK):
        X.append(X_all[i:i+LOOKBACK])
        y.append(X_all[i+LOOKBACK, close_idx])

    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer=Adam(LR), loss="mse")
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

print("Training model...")
model = build_model((LOOKBACK, X.shape[2]))
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# ===============================
# Recursive Forecast
# ===============================
print("Running recursive forecast...")
window = df[FEATURES].iloc[-LOOKBACK:].copy()
future_close = []

close_idx = FEATURES.index("Close")
close_min = scaler.data_min_[close_idx]
close_max = scaler.data_max_[close_idx]

for _ in range(FORECAST_DAYS):
    scaled = scaler.transform(window)
    pred_scaled = model.predict(scaled[np.newaxis, ...], verbose=0)[0, 0]
    pred_close = pred_scaled * (close_max - close_min) + close_min

    future_close.append(pred_close)

    next_row = window.iloc[-1].copy()
    next_row["Close"] = pred_close
    window = pd.concat([window.iloc[1:], next_row.to_frame().T])
    window = compute_indicators(window)[FEATURES]

future_dates = pd.bdate_range(
    start=df.index[-1] + BDay(1),
    periods=len(future_close)
)

# ===============================
# Plot
# ===============================
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(df.index[-120:], df["Close"].iloc[-120:], label="Historical", color="black")
ax.plot(future_dates, future_close, label="6M Forecast (Recursive)", color="tab:red")
ax.set_title("3481.TW | 6-Month Forecast (Recursive)", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
add_timestamp(ax)

out_path = os.path.join(
    RESULT_DIR,
    f"{datetime.now():%Y-%m-%d}_3481_6m_forecast_recursive.png"
)
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Saved → {out_path}")
