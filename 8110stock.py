# -*- coding: utf-8 -*-
"""
FireBase_Attention_LSTM_Direction.py  (3481stock.py)
- Attention-LSTM
- Multi-task: Return path + Direction + Trend(3-class)
- ✅ 群創（3481.TW）專屬調校版
  - 中大型面板股
  - 波動較低、趨勢延續性高
  - 6M Outlook 偏保守、不假噴
"""

import os, json, glob
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
from zoneinfo import ZoneInfo

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Softmax, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping

# ================= 時區 =================
now_tw = datetime.now(ZoneInfo("Asia/Taipei"))

# ================= Firebase =================
import firebase_admin
from firebase_admin import credentials, firestore

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
def load_df_from_firestore(
    ticker,
    collection="NEW_stock_data_liteon",
    days=600
):
    if db is None:
        raise ValueError("❌ Firestore 未初始化")

    rows = []
    for doc in db.collection(collection).stream():
        p = doc.to_dict().get(ticker)
        if p:
            rows.append({"date": doc.id, **p})

    if not rows:
        raise ValueError("⚠️ Firestore 無資料")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values("date")
          .tail(days)
          .set_index("date")
    )
    return df


# ================= 非交易日補 row =================
def ensure_latest_trading_row(df):
    today = pd.Timestamp(datetime.now().date())
    last = df.index.max()

    if last.normalize() >= today:
        return df

    all_days = pd.bdate_range(last, today)
    for d in all_days[1:]:
        if d not in df.index:
            df.loc[d] = df.loc[last]

    return df.sort_index()


def get_asof_trading_day(df):
    today = pd.Timestamp(datetime.now().date())
    last_trading = df.index.max()
    if last_trading.normalize() == today:
        return last_trading, True
    return last_trading, False


# ================= Feature Engineering（群創適用） =================
def add_features(df):
    df = df.copy()

    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].astype(float))

    # 均線（畫圖用）
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()

    # 波動 / 跳空
    df["HL_RANGE"] = (df["High"] - df["Low"]) / df["Close"]
    df["GAP"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # 量能相對
    df["VOL_REL"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

    # 波動尺度
    close = df["Close"].astype(float)
    df["RET_STD_20"] = np.log(close).diff().rolling(20).std()

    # 趨勢狀態
    ma60 = df["Close"].rolling(60)
    df["TREND_60"] = (df["Close"] - ma60.mean()) / (ma60.std() + 1e-9)
    df["TREND_SLOPE_20"] = df["Close"].rolling(20).mean().diff() / df["Close"]

    return df


# ================= Sequence =================
def create_sequences(
    df, features,
    steps=5, window=50,
    trend_h=20,
    k_flat=0.9,
    eps=1e-9
):
    X, y_ret, y_dir, y_tr3, idx = [], [], [], [], []

    close = df["Close"].astype(float)
    logret = np.log(close).diff()
    feat = df[features].values

    max_h = max(steps, trend_h)

    for i in range(window, len(df) - max_h):
        x = feat[i - window:i]
        if np.any(np.isnan(x)):
            continue

        scale = df["RET_STD_20"].iloc[i - 1]
        if not np.isfinite(scale) or scale < eps:
            continue

        future5 = logret.iloc[i:i + steps].values
        future20 = logret.iloc[i:i + trend_h].values
        if np.any(np.isnan(future5)) or np.any(np.isnan(future20)):
            continue

        y_ret.append(future5 / scale)
        y_dir.append(1.0 if future5.sum() > 0 else 0.0)

        cum = future20.sum()
        thr = k_flat * scale * np.sqrt(trend_h)

        cls = 2 if cum > thr else 0 if cum < -thr else 1
        onehot = np.zeros(3)
        onehot[cls] = 1.0

        X.append(x)
        y_tr3.append(onehot)
        idx.append(df.index[i])

    return (
        np.array(X),
        np.array(y_ret),
        np.array(y_dir),
        np.array(y_tr3),
        np.array(idx)
    )


# ================= Attention LSTM =================
def build_attention_lstm(
    input_shape,
    steps,
    max_daily_normret=2.2,
    learning_rate=5e-4,
    lstm_units=64
):
    inp = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=True)(inp)
    x = Dropout(0.2)(x)

    score = Dense(1)(x)
    w = Softmax(axis=1)(score)
    ctx = Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1))([x, w])

    raw = Dense(steps, activation="tanh")(ctx)
    out_ret = Lambda(lambda t: t * max_daily_normret, name="return")(raw)
    out_dir = Dense(1, activation="sigmoid", name="direction")(ctx)
    out_tr3 = Dense(3, activation="softmax", name="trend3")(ctx)

    model = Model(inp, [out_ret, out_dir, out_tr3])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={
            "return": tf.keras.losses.Huber(),
            "direction": "binary_crossentropy",
            "trend3": "categorical_crossentropy"
        },
        loss_weights={
            "return": 1.0,
            "direction": 0.4,
            "trend3": 1.3
        }
    )
    return model


# ================= 工具 =================
def last_valid_value(df, col, lookback=30):
    if col not in df.columns:
        return None
    s = df[col].iloc[-lookback:].dropna()
    return None if s.empty else float(s.iloc[-1])


# ================= Main =================
if __name__ == "__main__":
    TICKER = "3481.TW"
    COLLECTION = "NEW_stock_data_liteon"

    LOOKBACK = 50
    STEPS = 5
    TREND_H = 20
    K_FLAT = 0.9

    FEATURES = [
        "Close", "Open", "High", "Low",
        "Volume", "RSI", "MACD", "K", "D", "ATR_14",
        "HL_RANGE", "GAP", "VOL_REL",
        "TREND_60", "TREND_SLOPE_20"
    ]

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    MODEL_PATH = f"models/{TICKER}_attn_lstm.keras"

    df = load_df_from_firestore(TICKER, COLLECTION)
    df = ensure_latest_trading_row(df)
    df = add_features(df)
    df = df.dropna()

    X, y_ret, y_dir, y_tr3, idx = create_sequences(
        df, FEATURES,
        steps=STEPS,
        window=LOOKBACK,
        trend_h=TREND_H,
        k_flat=K_FLAT
    )

    split = int(len(X) * 0.85)
    X_tr, X_va = X[:split], X[split:]
    y_ret_tr, y_ret_va = y_ret[:split], y_ret[split:]
    y_dir_tr, y_dir_va = y_dir[:split], y_dir[split:]
    y_tr3_tr, y_tr3_va = y_tr3[:split], y_tr3[split:]
    idx_tr = idx[:split]

    scaler = MinMaxScaler()
    scaler.fit(df.loc[:idx_tr[-1], FEATURES].values)

    def scale_X(Xb):
        n, t, f = Xb.shape
        return scaler.transform(Xb.reshape(-1, f)).reshape(n, t, f)

    X_tr_s = scale_X(X_tr)
    X_va_s = scale_X(X_va)

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = build_attention_lstm(
            (LOOKBACK, len(FEATURES)),
            STEPS
        )

    model.fit(
        X_tr_s,
        {"return": y_ret_tr, "direction": y_dir_tr, "trend3": y_tr3_tr},
        validation_data=(X_va_s, {
            "return": y_ret_va,
            "direction": y_dir_va,
            "trend3": y_tr3_va
        }),
        epochs=120,
        batch_size=16,
        verbose=2,
        callbacks=[EarlyStopping(patience=12, restore_best_weights=True)]
    )

    model.save(MODEL_PATH)

    pred_ret, pred_dir, pred_tr3 = model.predict(X_va_s, verbose=0)

    raw_norm_returns = pred_ret[-1]
    p_dir = float(pred_dir[-1][0])
    p_tr = pred_tr3[-1]

    asof_date, _ = get_asof_trading_day(df)
    last_close = float(df.loc[asof_date, "Close"])
    scale_last = float(df.loc[asof_date, "RET_STD_20"])

    prices = []
    price = last_close
    for r in raw_norm_returns:
        price *= np.exp(float(r) * scale_last)
        prices.append(price)

    future_df = pd.DataFrame({
        "date": pd.bdate_range(asof_date + BDay(1), periods=STEPS),
        "Pred_Close": prices
    })

    out_csv = f"results/{datetime.now():%Y-%m-%d}_{TICKER}_forecast.csv"
    future_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("===================================")
    print(f"📈 3481.TW 5D 看漲機率：{p_dir:.2%}")
    print(f"📌 20D 趨勢：Down/Flat/Up = {p_tr}")
    print(f"💾 已輸出：{out_csv}")
