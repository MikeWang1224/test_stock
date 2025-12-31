# -*- coding: utf-8 -*-
"""
3481.TW 6M Outlook Only
- Attention-LSTM framework 精簡版
- 輸入 Firebase 近一年資料
- 特徵工程 + 6個月趨勢預測 + 圖片輸出
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Softmax, Lambda

from zoneinfo import ZoneInfo
now_tw = datetime.now(ZoneInfo("Asia/Taipei"))

# ---------------- Firebase ----------------
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


def load_df_from_firestore(ticker, collection="NEW_stock_data_liteon", days=500):
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
    df = df.sort_values("date").tail(days).set_index("date")
    return df

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

def add_features(df):
    df = df.copy()
    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].astype(float))
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    if all(c in df.columns for c in ["Open","High","Low","Close"]):
        df["HL_RANGE"] = (df["High"].astype(float)-df["Low"].astype(float))/df["Close"].astype(float)
        df["GAP"] = (df["Open"].astype(float)-df["Close"].shift(1).astype(float))/df["Close"].shift(1).astype(float)
    else:
        df["HL_RANGE"]=np.nan; df["GAP"]=np.nan
    df["VOL_REL"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["RET_STD_20"] = np.log(df["Close"].astype(float)).diff().rolling(20).std()
    ma60 = df["Close"].rolling(60)
    df["TREND_60"] = (df["Close"]-ma60.mean())/(ma60.std()+1e-9)
    df["TREND_SLOPE_20"] = df["Close"].rolling(20).mean().diff()/df["Close"]
    return df

def last_valid_value(df, col, lookback=30):
    if col not in df.columns: return None
    s = df[col].iloc[-lookback:]
    s = s[s.notna()]
    if s.empty: return None
    return float(s.iloc[-1])

# ---------------- Model ----------------
def build_attention_lstm(input_shape, steps, max_daily_normret=3.0, lstm_units=64):
    inp = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    score = Dense(1, name="attn_score")(x)
    weights = Softmax(axis=1, name="attn_weights")(score)
    context = Lambda(lambda t: tf.reduce_sum(t[0]*t[1],axis=1), name="attn_context")([x,weights])
    raw = Dense(steps, activation="tanh")(context)
    out_ret = Lambda(lambda t:t*max_daily_normret,name="return")(raw)
    model = Model(inp, out_ret)
    model.compile(optimizer="adam", loss="mse")
    return model

# ---------------- 6M Outlook ----------------
def plot_6m_trend_advanced(df, last_close, raw_norm_returns, scale_last, ticker, asof_date):
    MONTHS = 6
    DPM = 21
    eps = 1e-9

    close = df["Close"].astype(float)
    ret = np.log(close + eps).diff()
    daily_drift = float(ret.ewm(span=60).mean().tail(20).mean())
    daily_drift = np.clip(daily_drift, -0.01, 0.01)
    atr_ratio = 0.05
    monthly_logret = np.clip(daily_drift * DPM, -0.18, 0.18)
    model_1m_price = float(last_close * np.exp(monthly_logret))

    trend = []
    p = float(last_close)
    for _ in range(MONTHS):
        p *= np.exp(monthly_logret)
        trend.append(p)
    trend = np.array(trend)

    prices = [float(last_close)]
    centers = [float(last_close)]

    for m in range(1, MONTHS + 1):
        center = float(model_1m_price * 0.6 + trend[m - 1] * 0.4)
        price = center * (1 + 0.05 * np.sin(2 * np.pi * m / 80))
        centers.append(center)
        prices.append(price)

    # ⭐ 這裡把 centers 轉成 np.array
    centers = np.array(centers)
    prices = np.array(prices)

    labels = ["Now"] + [(asof_date + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(1, MONTHS + 1)]
    plt.figure(figsize=(15, 7))
    plt.fill_between(range(MONTHS + 1), centers * 0.95, centers * 1.05, alpha=0.18, label="Expected Range")
    plt.plot(range(MONTHS + 1), prices, "r-o", linewidth=2.8, label="Projected Path")
    plt.scatter(0, prices[0], s=180, marker="*", label="Today")

    for i, p in enumerate(prices[1:]):
        plt.text(i + 1, p, f"{p:.2f}", ha="center", fontsize=12)

    plt.xticks(range(MONTHS + 1), labels, fontsize=13)
    plt.title(f"{ticker} · 6M Outlook (Simplified)")
    plt.grid(alpha=0.3)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    out = f"results/{datetime.now():%Y-%m-%d}_{ticker}_6m.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 6M Outlook saved: {out}")


# ---------------- Main ----------------
if __name__=="__main__":
    TICKER="3481.TW"
    COLLECTION="NEW_stock_data_liteon"

    df=load_df_from_firestore(TICKER, COLLECTION, days=500)
    df=ensure_latest_trading_row(df)
    df=add_features(df)
    df=df.dropna()

    asof_date=df.index.max()
    last_close=float(df.loc[asof_date,"Close"])
    scale_last=float(df.loc[asof_date,"RET_STD_20"])
    scale_last=max(scale_last,1e-6)

    # 模型假設 output (normalized return) 5日平均作 6M drift 轉換
    raw_norm_returns=np.array([0.0,0.0,0.0,0.0,0.0])

    plot_6m_trend_advanced(df,last_close,raw_norm_returns,scale_last,TICKER,asof_date)
