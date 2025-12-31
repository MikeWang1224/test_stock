# -*- coding: utf-8 -*-
"""
3481.TW 群創光電
6-Month Forecast ONLY
- 方法與 8110 完全一致
- 完全使用 Firebase 資料
- 只輸出六個月預測圖
- 左下角顯示時間戳記（UTC+8）
"""

# ===============================
# Imports
# ===============================
import os
import json
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
STEPS = 120          # 約 6 個月交易日
EPOCHS = 60
BATCH = 32
LR = 5e-4

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
    all_days = pd.bdate_range(last, today)
    for d in all_days[1:]:
        if d not in df.index:
            df.loc[d] = df.loc[last]
    return df.sort_index()

def compute_indicators(df):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

    df["RET"] = df["Close"].pct_change()
    return df.dropna()

def make_dataset(df, features):
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(df[features])

    X, y = [], []
    close_idx = features.index("Close")

    for i in range(len(X_all) - LOOKBACK - STEPS):
        X.append(X_all[i:i+LOOKBACK])
        y.append(X_all[i+LOOKBACK:i+LOOKBACK+STEPS, close_idx])

    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(STEPS)
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

FEATURES = ["Close", "RSI", "MACD", "Volume"]
X, y, scaler = make_dataset(df, FEATURES)

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
# 6M Forecast
# ===============================
last_X = X[-1:]
pred_scaled = model.predict(last_X)[0]

close_scaler = MinMaxScaler()
close_scaler.fit(df[["Close"]])

future_close = close_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

future_dates = pd.bdate_range(
    start=df.index[-1] + BDay(1),
    periods=len(future_close)
)

# ===============================
# Plot
# ===============================
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(df.index[-120:], df["Close"].iloc[-120:], label="Historical", color="black")
ax.plot(future_dates, future_close, label="6-Month Forecast", color="tab:blue")

ax.set_title("3481.TW | 6-Month Forecast", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

add_timestamp(ax)

out_path = os.path.join(
    RESULT_DIR,
    f"{datetime.now():%Y-%m-%d}_3481_6m_forecast.png"
)

plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Saved → {out_path}")
