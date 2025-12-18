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

✅ 新增：同時輸出 PNG + CSV
  - results/YYYY-MM-DD_pred.png
  - results/YYYY-MM-DD_forecast.csv
  - results/YYYY-MM-DD_backtest.png
  - results/YYYY-MM-DD_backtest.csv
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

from datetime import datetime
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
def ensure_today_row(df):
    today = pd.Timestamp(datetime.now().date())
    last_date = df.index.max()
    if last_date < today:
        df.loc[today] = df.loc[last_date]
        print(f"⚠️ 今日無資料，使用 {last_date.date()} 補今日")
    return df.sort_index()

# ================= Feature Engineering =================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # ✅ Volume 尺度穩定（非常建議）
    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].astype(float))

    # 圖表用均線（保持不變）
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    return df

# ================= Sequence（避免錯位，且不亂切 df） =================
def create_sequences(df, features, steps=5, window=40):
    """
    X: t-window ~ t-1
    y_ret: t ~ t+steps-1 的 log return
    y_dir: 未來 steps 天累積方向
    idx: 每個樣本對應的「t 當天日期」（用來避免 scaler/split 座標系錯位）
    """
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
        idx.append(df.index[i])  # ✅ 這個樣本對應的 t 日期

    return np.array(X), np.array(y_ret), np.array(y_dir), np.array(idx)

# ================= Attention-LSTM（✅ return 限幅） =================
def build_attention_lstm(input_shape, steps, max_daily_logret=0.06):
    """
    max_daily_logret：限制單日 log-return 最大幅度，避免連乘價格爆炸
    常見範圍：0.04~0.08
    """
    inp = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)

    score = Dense(1, name="attn_score")(x)                 # (batch, time, 1)
    weights = Softmax(axis=1, name="attn_weights")(score)  # softmax over time
    context = Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1),
                     name="attn_context")([x, weights])    # (batch, hidden)

    # ✅ return head：tanh 限幅（結構性保證不會爆）
    raw = Dense(steps, activation="tanh")(context)          # [-1, 1]
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

# ================= 原預測圖（完全不動：新增 Today 標記） =================
def plot_and_save(df_hist, future_df):
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

    # ✅ Today 點與文字（hist 最後一個點）
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
    plt.savefig(f"results/{datetime.now():%Y-%m-%d}_pred.png",
                dpi=300, bbox_inches="tight")
    plt.close()
# ================= 回測決策分岔圖（PNG + CSV） =================
def plot_backtest_error(df):
    """
    決策式回測圖（Decision-based Backtest）

    特性：
    - 自動排除今天的 forecast
    - 使用最近一筆歷史 forecast
    - 不受 ensure_today_row() 假資料影響
    - 不怕週末 / 停市
    - 圖中加入 run timestamp，確保 Git 每次都會更新 PNG
    """

    today = pd.Timestamp(datetime.now().date())

    # ================= 找最近一次（排除今天）的 forecast =================
    if not os.path.exists("results"):
        print("⚠️ 無 results 資料夾，略過回測")
        return

    forecast_files = []
    for f in os.listdir("results"):
        if not f.endswith("_forecast.csv"):
            continue
        try:
            d = pd.to_datetime(f.split("_")[0])
        except Exception:
            continue

        if d < today:  # 明確排除今天
            forecast_files.append((d, f))

    if not forecast_files:
        print("⚠️ 找不到可用的歷史 forecast（已排除今天）")
        return

    forecast_files.sort(key=lambda x: x[0], reverse=True)
    forecast_date, forecast_name = forecast_files[0]
    forecast_csv = os.path.join("results", forecast_name)

    print(f"📄 Backtest 使用 forecast：{forecast_name}")

    future_df = pd.read_csv(forecast_csv, parse_dates=["date"])

    # ================= 決策日 t（最後一個真實交易日） =================
    valid_days = df.index[df.index < today]

    if len(valid_days) < 2:
        print("⚠️ 無足夠歷史交易日，略過回測")
        return

    t = valid_days[-1]
    t1 = t + BDay(1)

    # ================= 價格 =================
    close_t = float(df.loc[t, "Close"])
    pred_t1 = float(future_df.loc[0, "Pred_Close"])

    if t1 in df.index:
        actual_t1 = float(df.loc[t1, "Close"])
    else:
        actual_t1 = float(df["Close"].iloc[-1])

    # ================= 趨勢背景（三天） =================
    trend = df.loc[:t].tail(4)
    x_trend = np.arange(len(trend))
    x_t = x_trend[-1]

       # ================= 畫圖 =================
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    
    # 最近收盤趨勢
    ax.plot(
        x_trend,
        trend["Close"],
        "k-o",
        label="Recent Close"
    )
    
    # Pred 線
    ax.plot(
        [x_t, x_t + 1],
        [close_t, pred_t1],
        "r--o",
        linewidth=2.5,
        label="Pred (t → t+1)"
    )
    
    # Actual 線
    ax.plot(
        [x_t, x_t + 1],
        [close_t, actual_t1],
        "g-o",
        linewidth=2.5,
        label="Actual (t → t+1)"
    )
    
    # ================= 數值標註（全部統一在點右邊） =================
    dx = 0.08   
    price_offset = max(0.2, close_t * 0.002)# 或依股價調整，例如 0.2 ~ 0.5

    ax.text( 
        x_t,
        close_t + price_offset, 
        f"{close_t:.2f}", 
        ha="center", 
        va="bottom",   
        fontsize=18, 
        color="black" 
    )
    # Pred t+1
    ax.text(
        x_t + 1 + dx,
        pred_t1,
        f"Pred {pred_t1:.2f}",
        ha="left",
        va="center",
        fontsize=16,
        color="red"
    )
    
    # Actual t+1
    ax.text(
        x_t + 1 + dx,
        actual_t1,
        f"Actual {actual_t1:.2f}",
        ha="left",
        va="center",
        fontsize=16,
        color="green"
    )
    
    # ================= X 軸 =================
    labels = trend.index.strftime("%m-%d").tolist()
    labels.append(t1.strftime("%m-%d"))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    
    ax.set_title("2301.TW Decision Backtest (t → t+1)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ================= Run timestamp =================
    ax.text(
        0.01, 0.01,
        f"Generated at {now_tw:%Y-%m-%d %H:%M:%S} (TW)",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.4,
        ha="left",
        va="bottom"
    )



    # ================= 儲存 =================
    os.makedirs("results", exist_ok=True)
    print(f"🖼️ 儲存 backtest 圖：{today:%Y-%m-%d}_backtest.png")

    plt.savefig(
        f"results/{today:%Y-%m-%d}_backtest.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ================= CSV（單筆決策） =================
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
        f"results/{today:%Y-%m-%d}_backtest.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ================= CSV（單筆決策） =================
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
        f"results/{today:%Y-%m-%d}_backtest.csv",
        index=False,
        encoding="utf-8-sig"
    )

# ================= Main =================
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 40
    STEPS = 5

    df = load_df_from_firestore(TICKER, days=500)
    df = ensure_today_row(df)
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

    # ✅ scaler.fit 僅用 train 區間（用 idx_tr 的最後日期界定）
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
    raw_returns = pred_ret[-1]  # ✅ 已被結構性限幅

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

    # ✅ 預測數值輸出 CSV（隔天要疊今日實際用這份）
    os.makedirs("results", exist_ok=True)
    future_df.to_csv(f"results/{datetime.now():%Y-%m-%d}_forecast.csv",
                     index=False, encoding="utf-8-sig")

    plot_and_save(df, future_df)
    plot_backtest_error(df)
