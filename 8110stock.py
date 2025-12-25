# -*- coding: utf-8 -*-
"""
FireBase_Attention_LSTM_Direction.py  (8110stock.py)
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

✅ 新增：同時輸出 PNG + CSV（檔名含 ticker）
  - results/YYYY-MM-DD_TICKER_pred.png
  - results/YYYY-MM-DD_TICKER_forecast.csv
  - results/YYYY-MM-DD_TICKER_backtest.png
  - results/YYYY-MM-DD_TICKER_backtest.csv

✅ 華東 8110.TW 專屬強化（照前面建議改）
  A) Feature：加入 HL_RANGE / GAP / VOL_REL（更貼近中小型股/波動股）
  B) Target：預測「波動標準化」log-return（用 t-1 的 RET_STD_20 做尺度，避免偷看）
  C) 推回價格時：把預測的 normalized return 乘回 asof 的 RET_STD_20
  D) loss_weights：direction 權重提高（方向通常比精準價更可靠）
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
def ensure_today_row(df):
    today = pd.Timestamp(datetime.now().date())
    last_date = df.index.max()
    if last_date < today:
        df.loc[today] = df.loc[last_date]
        print(f"⚠️ 今日無資料，使用 {last_date.date()} 補今日")
    return df.sort_index()

# ================= Feature Engineering（華東專屬） =================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ✅ Volume 尺度穩定（非常建議）
    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].astype(float))

    # 圖表用均線（保持不變）
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()

    # ✅ 華東（波動/跳空/量能）特徵
    # 需要 Firestore 有 Open/High/Low（你 catch_stock.py 寫入是有的）
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        df["HL_RANGE"] = (df["High"].astype(float) - df["Low"].astype(float)) / df["Close"].astype(float)
        df["GAP"] = (df["Open"].astype(float) - df["Close"].shift(1).astype(float)) / df["Close"].shift(1).astype(float)
    else:
        # 若缺欄位，先給 NaN（後面 dropna 會排掉）
        df["HL_RANGE"] = np.nan
        df["GAP"] = np.nan

    # ✅ 量能相對強弱（用 log1p 之後的 Volume 去做比值即可）
    df["VOL_REL"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

    # ✅ 20日波動（用來標準化 y）
    close = df["Close"].astype(float)
    df["RET_STD_20"] = np.log(close).diff().rolling(20).std()

    return df

# ================= Sequence（標準化 return，避免波動 regime 影響） =================
def create_sequences(df, features, steps=5, window=40, eps=1e-9):
    """
    X: t-window ~ t-1
    y_ret: t ~ t+steps-1 的「normalized log return」
           normalized = logret / (RET_STD_20 at t-1)
           ✅ 用 t-1 的波動當尺度，避免偷看（no leakage）
    y_dir: 未來 steps 天累積方向（用 raw logret 判斷）
    idx: 每個樣本對應的「t 當天日期」
    """
    X, y_ret, y_dir, idx = [], [], [], []

    close = df["Close"].astype(float)
    logret = np.log(close).diff()

    if "RET_STD_20" not in df.columns:
        raise ValueError("⚠️ 缺少 RET_STD_20，請確認 add_features() 有被呼叫")

    feat = df[features].values

    for i in range(window, len(df) - steps):
        x_seq = feat[i - window:i]

        future_ret_raw = logret.iloc[i:i + steps].values
        if np.any(np.isnan(future_ret_raw)) or np.any(np.isnan(x_seq)):
            continue

        # ✅ 用 t-1 的波動尺度（避免偷看 t 的資訊）
        scale = df["RET_STD_20"].iloc[i - 1]
        if pd.isna(scale) or scale < eps:
            continue

        future_ret_norm = future_ret_raw / (float(scale) + eps)

        X.append(x_seq)
        y_ret.append(future_ret_norm)
        y_dir.append(1.0 if future_ret_raw.sum() > 0 else 0.0)
        idx.append(df.index[i])

    return np.array(X), np.array(y_ret), np.array(y_dir), np.array(idx)

# ================= Attention-LSTM（✅ return 限幅 + direction 權重提高） =================
def build_attention_lstm(input_shape, steps, max_daily_normret=3.0, learning_rate=6e-4, lstm_units=64):
    """
    max_daily_normret：限制「normalized」單日幅度，避免連乘爆炸
    - 因為現在 y 是 normalized return，所以限幅應該用「倍數」而不是 0.06 這種絕對值
    常見合理範圍：2 ~ 4
    """
    inp = Input(shape=input_shape)

    x = LSTM(lstm_units, return_sequences=True)(inp)
    x = Dropout(0.2)(x)

    score = Dense(1, name="attn_score")(x)
    weights = Softmax(axis=1, name="attn_weights")(score)
    context = Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1),
                     name="attn_context")([x, weights])

    # ✅ return head：tanh 限幅（normalized return）
    raw = Dense(steps, activation="tanh")(context)           # [-1, 1]
    out_ret = Lambda(lambda t: t * max_daily_normret, name="return")(raw)

    out_dir = Dense(1, activation="sigmoid", name="direction")(context)

    model = Model(inp, [out_ret, out_dir])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "return": tf.keras.losses.Huber(),
            "direction": "binary_crossentropy"
        },
        # ✅ 華東：方向更重要一些
        loss_weights={
            "return": 1.0,
            "direction": 0.8
        },
        metrics={
            "direction": [
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc")
            ]
        }
    )
    return model

# ================= 原預測圖（內容不動：新增 Today 標記） =================
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
    ax.set_title("2301.TW Attention-LSTM 預測")  # ✅ 內容不動

    os.makedirs("results", exist_ok=True)
    out_png = f"results/{datetime.now():%Y-%m-%d}_{ticker}_pred.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# ================= 回測決策分岔圖（PNG + CSV） =================
def plot_backtest_error(df, ticker: str):
    """
    決策式回測圖（Decision-based Backtest）
    - 自動排除今天的 forecast
    - 使用最近一筆歷史 forecast（同 ticker）
    """

    today = pd.Timestamp(datetime.now().date())

    if not os.path.exists("results"):
        print("⚠️ 無 results 資料夾，略過回測")
        return

    forecast_files = []
    for f in os.listdir("results"):
        if not f.endswith(f"_{ticker}_forecast.csv"):
            continue
        try:
            d = pd.to_datetime(f.split("_")[0])
        except Exception:
            continue
        if d < today:
            forecast_files.append((d, f))

    if not forecast_files:
        print("⚠️ 找不到可用的歷史 forecast（已排除今天 & 已限定 ticker）")
        return

    forecast_files.sort(key=lambda x: x[0], reverse=True)
    forecast_date, forecast_name = forecast_files[0]
    forecast_csv = os.path.join("results", forecast_name)

    print(f"📄 Backtest 使用 forecast：{forecast_name}")

    future_df = pd.read_csv(forecast_csv, parse_dates=["date"])

    valid_days = df.index[df.index < today]
    if len(valid_days) < 2:
        print("⚠️ 無足夠歷史交易日，略過回測")
        return

    t = valid_days[-1]
    t1 = t + BDay(1)

    close_t = float(df.loc[t, "Close"])
    pred_t1 = float(future_df.loc[0, "Pred_Close"])

    if t1 in df.index:
        actual_t1 = float(df.loc[t1, "Close"])
    else:
        actual_t1 = float(df["Close"].iloc[-1])

    trend = df.loc[:t].tail(4)
    x_trend = np.arange(len(trend))
    x_t = x_trend[-1]

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    ax.plot(x_trend, trend["Close"], "k-o", label="Recent Close")
    ax.plot([x_t, x_t + 1], [close_t, pred_t1], "r--o", linewidth=2.5, label="Pred (t → t+1)")
    ax.plot([x_t, x_t + 1], [close_t, actual_t1], "g-o", linewidth=2.5, label="Actual (t → t+1)")

    dx = 0.08
    price_offset = max(0.2, close_t * 0.002)

    ax.text(x_t, close_t + price_offset, f"{close_t:.2f}", ha="center", va="bottom", fontsize=18, color="black")
    ax.text(x_t + 1 + dx, pred_t1, f"Pred {pred_t1:.2f}", ha="left", va="center", fontsize=16, color="red")
    ax.text(x_t + 1 + dx, actual_t1, f"Actual {actual_t1:.2f}", ha="left", va="center", fontsize=16, color="green")

    labels = trend.index.strftime("%m-%d").tolist()
    labels.append(t1.strftime("%m-%d"))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)

    ax.set_title("2301.TW Decision Backtest (t → t+1)")  # ✅ 內容不動
    ax.legend()
    ax.grid(alpha=0.3)

    ax.text(
        0.01, 0.01,
        f"Generated at {now_tw:%Y-%m-%d %H:%M:%S} (TW)",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.4,
        ha="left",
        va="bottom"
    )

    os.makedirs("results", exist_ok=True)
    out_png = f"results/{today:%Y-%m-%d}_{ticker}_backtest.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    bt = pd.DataFrame([{
        "forecast_date": forecast_date.date(),
        "decision_day": t.date(),
        "close_t": close_t,
        "pred_t1": pred_t1,
        "actual_t1": actual_t1,
        "direction_pred": int(np.sign(pred_t1 - close_t)),
        "direction_actual": int(np.sign(actual_t1 - close_t))
    }])

    out_csv = f"results/{today:%Y-%m-%d}_{ticker}_backtest.csv"
    bt.to_csv(out_csv, index=False, encoding="utf-8-sig")

# ================= Main =================
if __name__ == "__main__":
    TICKER = "8110.TW"
    COLLECTION = "NEW_stock_data_liteon"

    # ✅ 華東專屬設定（normalized return 版本）
    STOCK_CONFIG = {
        "8110.TW": {
            "LOOKBACK": 40,
            "STEPS": 5,
            "MAX_DAILY_NORMRET": 3.0,  # normalized return 限幅（2~4 常見）
            "LR": 6e-4,
            "LSTM_UNITS": 64
        },
    }

    cfg = STOCK_CONFIG.get(TICKER, {
        "LOOKBACK": 40,
        "STEPS": 5,
        "MAX_DAILY_NORMRET": 3.0,
        "LR": 6e-4,
        "LSTM_UNITS": 64
    })

    LOOKBACK = cfg["LOOKBACK"]
    STEPS = cfg["STEPS"]

    os.makedirs("models", exist_ok=True)
    MODEL_PATH = f"models/{TICKER}_attn_lstm.keras"

    # ---------- Data ----------
    df = load_df_from_firestore(TICKER, collection=COLLECTION, days=500)
    df = ensure_today_row(df)
    df = add_features(df)

    # ✅ 華東專屬特徵（含 OHLC + 波動/跳空/量能）
    FEATURES = [
        "Close", "Open", "High", "Low",
        "Volume", "RSI", "MACD", "K", "D", "ATR_14",
        "HL_RANGE", "GAP", "VOL_REL"
    ]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"⚠️ Firestore 資料缺欄位：{missing}\n"
            f"請確認 catch_stock.py 寫回 8110.TW 時包含 Open/High/Low/Close/Volume，且指標欄位已寫入。"
        )

    # RET_STD_20 是 y 的尺度，需要一起存在（add_features 會做）
    if "RET_STD_20" not in df.columns:
        raise ValueError("⚠️ 缺少 RET_STD_20，請確認 add_features() 有被呼叫")

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

    # ---------- Model (專屬) ----------
    if os.path.exists(MODEL_PATH):
        print(f"✅ 載入既有模型：{MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=True)
    else:
        model = build_attention_lstm(
            (LOOKBACK, len(FEATURES)),
            STEPS,
            max_daily_normret=cfg["MAX_DAILY_NORMRET"],
            learning_rate=cfg["LR"],
            lstm_units=cfg["LSTM_UNITS"]
        )

    model.fit(
        X_tr_s,
        {"return": y_ret_tr, "direction": y_dir_tr},
        epochs=80,
        batch_size=16,
        verbose=2,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
    )

    model.save(MODEL_PATH)
    print(f"💾 已儲存模型：{MODEL_PATH}")

    pred_ret, pred_dir = model.predict(X_te_s, verbose=0)
    raw_norm_returns = pred_ret[-1]  # ✅ normalized returns（已限幅）

    print(f"📈 預測方向機率（看漲）: {pred_dir[-1][0]:.2%}")

    asof_date = df.index.max()
    last_close = float(df.loc[asof_date, "Close"])

    # ✅ 把 normalized return 乘回波動尺度（用 asof 的 RET_STD_20）
    scale_last = float(df.loc[asof_date, "RET_STD_20"])
    if not np.isfinite(scale_last) or scale_last <= 0:
        # fallback：用最近 20 天 std 估
        scale_last = float(np.log(df["Close"].astype(float)).diff().rolling(20).std().iloc[-1])
    scale_last = max(scale_last, 1e-6)

    prices = []
    price = last_close
    for r_norm in raw_norm_returns:
        r = float(r_norm) * scale_last
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

    # ✅ 預測數值輸出 CSV（檔名含 ticker）
    os.makedirs("results", exist_ok=True)
    forecast_csv = f"results/{datetime.now():%Y-%m-%d}_{TICKER}_forecast.csv"
    future_df.to_csv(forecast_csv, index=False, encoding="utf-8-sig")

    # ✅ 圖輸出（內容不動、檔名改含 ticker）
    plot_and_save(df, future_df, ticker=TICKER)
    plot_backtest_error(df, ticker=TICKER)
