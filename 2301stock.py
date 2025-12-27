# -*- coding: utf-8 -*-
"""
FireBase_Attention_LSTM_Direction.py
- Attention-LSTM
- Multi-task: Return path + Direction
- ✅ 小資料友善版：更穩、更不容易亂噴
  1) LOOKBACK=40, STEPS=5
  2) LSTM + Attention pooling（參數比 Transformer 更適合小資料）
  3) ✅ Return head 加 tanh 限幅（避免預測爆炸）
  4) ✅ Volume 做 log1p（小資料更穩）
- 圖表輸出完全不變（保留 Today 標記）

✅ 改1：修正 scaler fit / split 座標系，避免資料洩漏（leakage）
  - create_sequences 回傳每個樣本對應的日期 idx
  - split 用樣本數切，scaler.fit 只用 train 區間的 df 特徵

✅ 檔名改成含 ticker
  - results/YYYY-MM-DD_2301.TW_pred.png
  - results/YYYY-MM-DD_2301.TW_forecast.csv
  - results/YYYY-MM-DD_2301.TW_backtest.png
  - results/YYYY-MM-DD_2301.TW_backtest.csv
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Softmax, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping

from zoneinfo import ZoneInfo
now_tw = datetime.now(ZoneInfo("Asia/Taipei"))

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
def load_df_from_firestore(ticker, collection="NEW_stock_data_liteon", days=500):
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
def ensure_latest_trading_row(df):
    today = pd.Timestamp(datetime.now().date())
    last = df.index.max()

    # 補齊中間缺的 BDay（例如 12/25）
    all_days = pd.bdate_range(last, today)

    for d in all_days[1:]:
        if d not in df.index:
            df.loc[d] = df.loc[last]

    return df.sort_index()


# ================= Feature Engineering =================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].astype(float))

    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    return df

# ================= Sequence =================
def create_sequences(df, features, steps=5, window=40):
    X, y_ret, y_dir, idx = [], [], [], []

    close = df["Close"].astype(float)
    logret = np.log(close).diff()
    feat = df[features].values

    for i in range(window, len(df) - steps):
        x_seq = feat[i - window:i]
        future_ret = logret.iloc[i:i + steps].values
        if np.any(np.isnan(future_ret)) or np.any(np.isnan(x_seq)):
            continue
        X.append(x_seq)
        y_ret.append(future_ret)
        y_dir.append(1.0 if future_ret.sum() > 0 else 0.0)
        idx.append(df.index[i])

    return np.array(X), np.array(y_ret), np.array(y_dir), np.array(idx)

# ================= Attention-LSTM =================
def build_attention_lstm(input_shape, steps, max_daily_logret=0.06):
    inp = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)

    score = Dense(1, name="attn_score")(x)
    weights = Softmax(axis=1, name="attn_weights")(score)
    context = Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1),
                     name="attn_context")([x, weights])

    raw = Dense(steps, activation="tanh")(context)
    out_ret = Lambda(lambda t: t * max_daily_logret, name="return")(raw)

    out_dir = Dense(1, activation="sigmoid", name="direction")(context)

    model = Model(inp, [out_ret, out_dir])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7e-4),
        loss={
            "return": tf.keras.losses.Huber(),
            "direction": "binary_crossentropy"
        },
        loss_weights={
            "return": 1.0,
            "direction": 0.4
        },
        metrics={
            "direction": [tf.keras.metrics.BinaryAccuracy(name="acc"),
                          tf.keras.metrics.AUC(name="auc")]
        }
    )
    return model

# ================= 原預測圖（內容不動） =================
def plot_and_save(df_hist, future_df, ticker: str):
    hist = df_hist.tail(10)
    hist_dates = hist.index.strftime("%m-%d").tolist()
    future_dates = future_df["date"].dt.strftime("%m-%d").tolist()

    all_dates = hist_dates + future_dates
    x_hist = np.arange(len(hist_dates))
    x_future = np.arange(len(hist_dates), len(all_dates))

    plt.figure(figsize=(18,8))
    ax = plt.gca()

    ax.plot(x_hist, hist["Close"], label="Close")
    ax.plot(x_hist, hist["SMA5"], label="SMA5")
    ax.plot(x_hist, hist["SMA10"], label="SMA10")

    today_x = x_hist[-1]
    today_y = float(hist["Close"].iloc[-1])
    ax.scatter([today_x], [today_y], marker="*", s=160, label="Today Close")
    ax.text(today_x, today_y + 0.3, f"Today {today_y:.2f}",
            fontsize=17, ha="center")

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["Close"].iloc[-1]] + future_df["Pred_Close"].tolist(),
        "r:o", label="Pred Close"
    )

    for i, price in enumerate(future_df["Pred_Close"]):
        ax.text(x_future[i], price + 0.3, f"{price:.2f}",
                color="red", fontsize=15, ha="center")

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
    ax.set_xticklabels(all_dates, rotation=45, ha="right", fontsize=15)
    ax.legend()
    ax.set_title("2301.TW Attention-LSTM 預測")

    os.makedirs("results", exist_ok=True)
    out_png = f"results/{datetime.now():%Y-%m-%d}_{ticker}_pred.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# ================= 回測決策分岔圖（檔名含 ticker） =================
def plot_backtest_error(df, ticker):
    if not os.path.exists("results"):
        print("⚠️ 無 results 資料夾，略過回測")
        return

    # === 找最近一份「已發生」的 forecast ===
    suffix = f"_{ticker}_forecast.csv"
    forecast_files = []

    for f in os.listdir("results"):
        if not f.endswith(suffix):
            continue
        try:
            d = pd.to_datetime(f.split("_")[0])
            forecast_files.append((d, f))
        except Exception:
            continue

    if not forecast_files:
        print(f"⚠️ 找不到 forecast：{ticker}")
        return

    last_trading_day = df.index.max()
    valid = []
    
    for d, f in forecast_files:
        # forecast_date + 1 必須已經有實際資料
        idx = df.index.get_indexer([d])
        if idx[0] == -1:
            continue
        if idx[0] + 1 < len(df.index):
            valid.append((d, f))
    
    if not valid:
        print("⚠️ 找不到可回測的 forecast（尚無實際 t+1）")
        return
    
    # 用最近一個「已完成回測條件」的 forecast
    forecast_date, forecast_name = max(valid, key=lambda x: x[0])

    future_df = pd.read_csv(
        os.path.join("results", forecast_name),
        parse_dates=["date"]
    )

    # === 只用真實交易日 ===
    t, t1 = get_last_two_trading_days(df)

    close_t = float(df.loc[t, "Close"])
    actual_t1 = float(df.loc[t1, "Close"])

    # forecast 的第一天必須是 t1
    pred_row = future_df[future_df["date"] == t1]
    if pred_row.empty:
        print("⚠️ forecast 與交易日未對齊，略過回測")
        return

    pred_t1 = float(pred_row["Pred_Close"].iloc[0])

    # === 繪圖 ===
    trend = df.loc[:t].tail(4)
    x_trend = np.arange(len(trend))
    x_t = x_trend[-1]

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    ax.plot(x_trend, trend["Close"], "k-o", label="Recent Close")
    ax.plot([x_t, x_t + 1], [close_t, pred_t1],
            "r--o", linewidth=2.5, label="Pred (t → t+1)")
    ax.plot([x_t, x_t + 1], [close_t, actual_t1],
            "g-o", linewidth=2.5, label="Actual (t → t+1)")

    price_offset = max(0.2, close_t * 0.002)

    ax.text(x_t, close_t + price_offset, f"{close_t:.2f}",
            ha="center", fontsize=18)
    ax.text(x_t + 1.05, pred_t1, f"Pred {pred_t1:.2f}",
            color="red", fontsize=16, va="center")
    ax.text(x_t + 1.05, actual_t1, f"Actual {actual_t1:.2f}",
            color="green", fontsize=16, va="center")

    labels = trend.index.strftime("%m-%d").tolist()
    labels.append(t1.strftime("%m-%d"))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)

    ax.set_title(f"{ticker} Decision Backtest (t → t+1)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax.text(
        0.01, 0.01,
        f"Generated at {now_tw:%Y-%m-%d %H:%M:%S} (TW)",
        transform=ax.transAxes,
        fontsize=8, alpha=0.4
    )

    os.makedirs("results", exist_ok=True)
    today = datetime.now().date()
    plt.savefig(f"results/{today}_{ticker}_backtest.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # === CSV ===
    bt = pd.DataFrame([{
        "forecast_date": forecast_date.date(),
        "decision_day": t.date(),
        "close_t": close_t,
        "pred_t1": pred_t1,
        "actual_t1": actual_t1,
        "direction_pred": int(np.sign(pred_t1 - close_t)),
        "direction_actual": int(np.sign(actual_t1 - close_t))
    }])

    bt.to_csv(
        f"results/{today}_{ticker}_backtest.csv",
        index=False,
        encoding="utf-8-sig"
    )


def get_last_two_trading_days(df):
    """
    回傳最後兩個「真實交易日」 (t, t+1)
    """
    idx = df.index.sort_values()
    if len(idx) < 2:
        raise ValueError("⚠️ 交易日不足，無法回測")
    return idx[-2], idx[-1]


# ================= Main =================
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 40
    STEPS = 5

    df = load_df_from_firestore(TICKER, days=500)
    df = ensure_latest_trading_row(df)
    df = add_features(df)

    FEATURES = ["Close", "Volume", "RSI", "MACD", "K", "D", "ATR_14"]
    df = df.dropna()

    X, y_ret, y_dir, idx = create_sequences(df, FEATURES, steps=STEPS, window=LOOKBACK)
    print(f"df rows: {len(df)} | X samples: {len(X)}")

    if len(X) < 40:
        raise ValueError("⚠️ 可用序列太少（<40）。建議：降低 LOOKBACK/STEPS 或檢查資料是否缺欄位/過多 NaN。")

    split = int(len(X) * 0.85)

    X_tr, X_te = X[:split], X[split:]
    y_ret_tr, y_ret_te = y_ret[:split], y_ret[split:]
    y_dir_tr, y_dir_te = y_dir[:split], y_dir[split:]
    idx_tr, idx_te = idx[:split], idx[split:]

    train_end_date = pd.Timestamp(idx_tr[-1])
    df_for_scaler = df.loc[:train_end_date, FEATURES].copy()

    if len(df_for_scaler) < LOOKBACK + 5:
        raise ValueError("⚠️ train 區間太短，無法穩定 fit scaler。請確認資料量或調整 LOOKBACK。")

    sx = MinMaxScaler()
    sx.fit(df_for_scaler.values)

    def scale_X(Xb):
        n, t, f = Xb.shape
        return sx.transform(Xb.reshape(-1, f)).reshape(n, t, f)

    X_tr_s = scale_X(X_tr)
    X_te_s = scale_X(X_te)

    model = build_attention_lstm(
        (LOOKBACK, len(FEATURES)),
        STEPS,
        max_daily_logret=0.06
    )

    model.fit(
        X_tr_s,
        {"return": y_ret_tr, "direction": y_dir_tr},
        epochs=80,
        batch_size=16,
        verbose=2,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
    )

    pred_ret, pred_dir = model.predict(X_te_s, verbose=0)
    raw_returns = pred_ret[-1]

    print(f"📈 預測方向機率（看漲）: {pred_dir[-1][0]:.2%}")

    asof_date = df.index.max()
    last_close = float(df.loc[asof_date, "Close"])

    prices = []
    price = last_close
    for r in raw_returns:
        price *= np.exp(r)
        prices.append(price)

    seq = df.loc[:asof_date, "Close"].iloc[-10:].tolist()
    future = []
    for p in prices:
        seq.append(p)
        future.append({
            "Pred_Close": float(p),
            "Pred_MA5": float(np.mean(seq[-5:])),
            "Pred_MA10": float(np.mean(seq[-10:]))
        })

    future_df = pd.DataFrame(future)
    future_df["date"] = pd.bdate_range(
        start=df.index.max() + BDay(1),
        periods=STEPS
    )

    os.makedirs("results", exist_ok=True)
    forecast_csv = f"results/{datetime.now():%Y-%m-%d}_{TICKER}_forecast.csv"
    future_df.to_csv(forecast_csv, index=False, encoding="utf-8-sig")

    plot_and_save(df, future_df, ticker=TICKER)
    plot_backtest_error(df, ticker=TICKER)
