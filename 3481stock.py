# -*- coding: utf-8 -*-
"""
3481.TW 群創光電
6-Month Forecast ONLY
- 方法與原本一致（LSTM 多步）
- 只輸出六個月預測圖
- 左下角顯示時間戳記（UTC+8）
"""

# ===============================
# Imports
# ===============================
import os
import numpy as np
import pandas as pd
import yfinance as yf
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
# Config
# ===============================
TICKER = "3481.TW"
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
def compute_indicators(df):
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
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
        future_close = X_all[i+LOOKBACK:i+LOOKBACK+STEPS, close_idx]
        y.append(future_close)

    return np.array(X), np.array(y), scaler


def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(STEPS)
    ])
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
print("Downloading data...")
df = yf.download(TICKER, period="5y", auto_adjust=False)

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

# inverse scale Close only
close_scaler = MinMaxScaler()
close_scaler.fit(df[["Close"]])
future_close = close_scaler.inverse_transform(
    pred_scaled.reshape(-1, 1)
).flatten()

start_date = df.index[-1]
future_dates = pd.bdate_range(
    start=start_date + BDay(1),
    periods=len(future_close)
)

# ===============================
# Plot (ONLY 6M)
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
    label="6-Month Forecast",
    color="tab:blue"
)

ax.set_title("3481.TW | 6-Month Forecast", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# ✅ 左下角時間戳
add_timestamp(ax)

out_path = os.path.join(
    RESULT_DIR,
    f"{datetime.now():%Y-%m-%d}_3481_6m_forecast.png"
)

plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Saved → {out_path}")
