#8110stock 


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
def load_df_from_firestore(
    ticker,
    collection="NEW_stock_data_liteon",
    days=500
):
    if db is None:
        raise ValueError("❌ Firestore 未初始化")

    rows = []

    for doc in db.collection(collection).stream():
        p = doc.to_dict().get(ticker)
        if p:
            rows.append({
                "date": doc.id,   # YYYY-MM-DD
                **p
            })

    if not rows:
        raise ValueError("⚠️ Firestore 無資料")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # ✅ 這裡才是「防假日的第一道門」
    df = (
        df.sort_values("date")
          .tail(days)          # 只保留最近 N 筆「交易日」
          .set_index("date")
    )
    return df



# ================= 假日補今天 =================
def ensure_latest_trading_row(df):
    """
    若今天是非交易日，補 row（forward fill）
    但 Close 不會變，用於「預測 today+1」
    """
    today = pd.Timestamp(datetime.now().date())
    last = df.index.max()

    if last.normalize() >= today:
        return df

    all_days = pd.bdate_range(last, today)

    for d in all_days[1:]:
        if d not in df.index:
            df.loc[d] = df.loc[last]

    return df.sort_index()


def get_asof_trading_day(df: pd.DataFrame):
    """
    回傳 (asof_date, is_today_trading)
    - 若今天是交易日 → 用今天
    - 若今天非交易日 → 用最近一個交易日
    """
    today = pd.Timestamp(datetime.now().date())
    last_trading_day = df.index.max()

    if last_trading_day.normalize() == today:
        return last_trading_day, True
    else:
        return last_trading_day, False



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

    # 🔧 ADD: Regime / 波段狀態特徵（不存 Firebase）
    ma60 = df["Close"].rolling(60)
    df["TREND_60"] = (df["Close"] - ma60.mean()) / (ma60.std() + 1e-9)
    
    df["TREND_SLOPE_20"] = (
        df["Close"].rolling(20).mean().diff()
    ) / df["Close"]
    
    return df


# ================= Sequence（標準化 return，避免波動 regime 影響） =================
def create_sequences(
    df, features,
    steps=5, window=40,
    trend_h=20,           # ✅ 新增：趨勢 horizon（交易日）
    k_flat=0.8,           # ✅ 新增：盤整門檻（越大越保守）
    eps=1e-9
):
    """
    X: t-window ~ t-1
    y_ret: t ~ t+steps-1 normalized log return (用 t-1 波動做尺度)
    y_dir: 未來 steps 天累積方向（二分類，保留給短線）
    y_trend3: 未來 trend_h 天趨勢（三分類 Up/Flat/Down）✅ 更貼近真實
      - 用波動門檻：|cumret| < k_flat * scale * sqrt(trend_h) => Flat
    idx: 每個樣本對應 t 當天日期
    """
    X, y_ret, y_dir, y_trend3, idx = [], [], [], [], []

    close = df["Close"].astype(float)
    logret = np.log(close).diff()

    if "RET_STD_20" not in df.columns:
        raise ValueError("⚠️ 缺少 RET_STD_20，請確認 add_features() 有被呼叫")

    feat = df[features].values

    # 需要同時滿足 steps 與 trend_h 的未來資料
    max_h = max(steps, trend_h)

    for i in range(window, len(df) - max_h):
        x_seq = feat[i - window:i]
        if np.any(np.isnan(x_seq)):
            continue

        # ✅ 用 t-1 波動尺度（避免偷看）
        scale = df["RET_STD_20"].iloc[i - 1]
        if pd.isna(scale) or scale < eps:
            continue
        scale = float(scale) + eps

        # ---------- 5D return head ----------
        future_ret_raw_5d = logret.iloc[i:i + steps].values
        if np.any(np.isnan(future_ret_raw_5d)):
            continue
        future_ret_norm_5d = future_ret_raw_5d / scale

        # 短線方向（二分類，保留）
        dir_5d = 1.0 if future_ret_raw_5d.sum() > 0 else 0.0

        # ---------- 20D trend head (3-class) ----------
        future_ret_raw_tr = logret.iloc[i:i + trend_h].values
        if np.any(np.isnan(future_ret_raw_tr)):
            continue

        cum = float(future_ret_raw_tr.sum())  # log累積
        # ✅ 盤整門檻：波動 * sqrt(h)
        thr = float(k_flat) * scale * np.sqrt(float(trend_h))

        # class: 0=Down, 1=Flat, 2=Up
        if cum > thr:
            cls = 2
        elif cum < -thr:
            cls = 0
        else:
            cls = 1

        onehot = np.zeros(3, dtype=np.float32)
        onehot[cls] = 1.0

        X.append(x_seq)
        y_ret.append(future_ret_norm_5d)
        y_dir.append(dir_5d)
        y_trend3.append(onehot)
        idx.append(df.index[i])

    return (
        np.array(X),
        np.array(y_ret),
        np.array(y_dir),
        np.array(y_trend3),
        np.array(idx)
    )

def build_attention_lstm(
    input_shape,
    steps,
    max_daily_normret=3.0,
    learning_rate=6e-4,
    lstm_units=64
):
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

    # ✅ 5D direction（短線）
    out_dir = Dense(1, activation="sigmoid", name="direction")(context)

    # ✅ 20D trend（三分類：Down/Flat/Up）→ 更貼近真實
    out_trend = Dense(3, activation="softmax", name="trend3")(context)

    model = Model(inp, [out_ret, out_dir, out_trend])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "return": tf.keras.losses.Huber(),
            "direction": "binary_crossentropy",
            "trend3": "categorical_crossentropy"
        },
        # ✅ 趨勢比短線方向更重要（更貼近真實）
        loss_weights={
            "return": 1.0,
            "direction": 0.4,
            "trend3": 1.2
        },
        metrics={
            "direction": [
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc")
            ],
            "trend3": [
                tf.keras.metrics.CategoricalAccuracy(name="acc")
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
        # === 只保留真實交易日（排除 ensure_latest_trading_row 補的假日）===
    real_df = df.copy()
    real_df = real_df[real_df["Close"].diff().abs() > 1e-9]

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

        # === 用「真實交易日」決定 t / t+1 ===
    valid_days = real_df.index[real_df.index < today]

    if len(valid_days) < 2:
        print("⚠️ 無足夠真實交易日，略過回測")
        return

    # t = 最後一個可決策日
    # t1 = 真正發生的下一個交易日
    t = valid_days[-2]
    t1 = valid_days[-1]


    close_t = float(real_df.loc[t, "Close"])
    pred_t1 = float(future_df.loc[0, "Pred_Close"])
    actual_t1 = float(real_df.loc[t1, "Close"])
    

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

import glob

def plot_6m_trend_advanced(
    df: pd.DataFrame,
    last_close: float,
    raw_norm_returns: np.ndarray,
    scale_last: float,
    ticker: str,
    asof_date: pd.Timestamp,
    amp: float = 1.0,
    pred_ret_all=None,          # 可選：傳入 pred_ret 全部 (N, STEPS)
    pred_dir_last=None,         # 可選：傳入最後一筆方向機率 (float, 0~1)
    k_ens: int = 20
):
    """
    8110-tuned 6M Outlook (更貼近現實版)
    - 不再把 5日預測硬複利 21次 → 改成「模型只調 drift 的 edge」
    - 用 pred_dir 信心調整模型影響力（不確定就回歸保守）
    - Expected Range 用歷史 backtest 誤差校準（有檔就用，沒有就 fallback ATR）
    """
    MONTHS = 6
    DPM = 21
    eps = 1e-9

    # -----------------------------
    # 0) 取模型 5日輸出（ensemble 更穩）
    # -----------------------------
    if pred_ret_all is not None:
        try:
            K = min(int(k_ens), len(pred_ret_all))
            base5 = np.median(np.asarray(pred_ret_all)[-K:], axis=0).astype(float)
        except Exception:
            base5 = np.array(raw_norm_returns, dtype=float)
    else:
        base5 = np.array(raw_norm_returns, dtype=float)

    if base5 is None or len(base5) == 0:
        raise ValueError("❌ base5 為空：raw_norm_returns/pred_ret_all 無法使用")

    # -----------------------------
    # 1) 先算歷史 drift（用 log-return 更穩）
    # -----------------------------
    close = df["Close"].astype(float)
    logp = np.log(close + eps)
    ret = logp.diff()

    daily_drift = float(ret.ewm(span=60).mean().tail(20).mean())
    daily_drift = float(np.clip(daily_drift, -0.01, 0.01))  # 防爆

    # -----------------------------
    # 2) Regime：RSI/ATR → trend_score（控制 drift）
    # -----------------------------
    atr = last_valid_value(df, "ATR_14", lookback=40)
    rsi = last_valid_value(df, "RSI", lookback=40)

    if atr is None:
        raise ValueError("❌ 無可用 ATR_14（最近 40 日皆為 NaN）")

    atr_ratio = float(atr) / float(last_close + eps)
    vol_regime = atr_ratio

    trend_score = 1.0
    if rsi is not None and rsi > 75:
        trend_score *= 0.35
    elif rsi is not None and rsi > 65:
        trend_score *= 0.65

    if vol_regime < 0.015:
        trend_score *= 0.6
    if vol_regime > 0.08:
        trend_score *= 0.75

    # -----------------------------
    # 3) ✅ 模型 edge：只用來「調 drift」，不再 21 次複利推 1M
    # -----------------------------
    # base5 是 normalized return → 乘回 scale_last 變成日 log-return edge
    edge_daily = float(np.mean(base5)) * float(scale_last)

    # amp 不要直接放大 edge（避免噴）；只對週期震盪可放大
    # RSI 過熱時，edge 再壓一點（更貼近現實）
    if rsi is not None and rsi > 75:
        edge_daily *= 0.6

    # ✅ cap：8110 單日 edge ±0.4% 已經很寬
    edge_daily = float(np.clip(edge_daily, -0.004, 0.004))

    # 最終 drift（加上模型 edge，再乘 trend_score）
    daily_drift_adj = (daily_drift + edge_daily) * trend_score
    daily_drift_adj = float(np.clip(daily_drift_adj, -0.01, 0.01))
    monthly_logret = daily_drift_adj * DPM

    # 1M anchor（更穩、更像現實）
    model_1m_price = float(last_close * np.exp(monthly_logret))

    # -----------------------------
    # 4) FFT 週期：用 log-return 做（避免趨勢被當週期）
    # -----------------------------
    r = ret.dropna().iloc[-180:].values
    if len(r) < 60:
        cycle_p = 80
    else:
        r_centered = r - r.mean()
        fft_p = np.fft.rfft(r_centered)
        freq_p = np.fft.rfftfreq(len(r_centered), d=1)
        mag = np.abs(fft_p)
        mag[0] = 0.0
        idx_p = int(np.argmax(mag))
        if idx_p == 0 or freq_p[idx_p] <= 1e-6:
            cycle_p = 80
        else:
            cycle_p = int(round(1 / freq_p[idx_p]))
            cycle_p = int(np.clip(cycle_p, 40, 120))

    vol_series = df["Volume"].iloc[-180:].dropna().astype(float).values
    if len(vol_series) < 60:
        cycle_v = 30
    else:
        v_centered = vol_series - vol_series.mean()
        fft_v = np.fft.rfft(v_centered)
        freq_v = np.fft.rfftfreq(len(v_centered), d=1)
        magv = np.abs(fft_v)
        magv[0] = 0.0
        idx_v = int(np.argmax(magv))
        if idx_v == 0 or freq_v[idx_v] <= 1e-6:
            cycle_v = 35
        else:
            cycle_v = int(round(1 / freq_v[idx_v]))
            cycle_v = int(np.clip(cycle_v, 20, 60))

    # -----------------------------
    # 5) 震盪幅度 base_amp（由 ATR%，RSI過熱不要放大）
    # -----------------------------
    if rsi is None:
        rsi = 50.0
    rsi_strength = abs(float(rsi) - 50.0) / 50.0
    rsi_factor = np.clip(0.6 + 0.8 * rsi_strength, 0.7, 1.25)
    if rsi > 75:
        rsi_factor *= 0.75  # 過熱壓縮震盪

    base_amp = float(np.clip(atr_ratio * rsi_factor, 0.02, 0.18))
    # ✅ amp 只用來調週期震盪（不是調模型 drift）
    base_amp = float(np.clip(base_amp * float(amp), 0.02, 0.22))

    # -----------------------------
    # 6) drift 基準線（不用覆蓋 trend[0]，避免雙重 anchor）
    # -----------------------------
    trend = []
    p = float(last_close)
    for _ in range(MONTHS):
        p *= np.exp(monthly_logret)
        trend.append(p)
    trend = np.array(trend, dtype=float)

    # -----------------------------
    # 7) ✅ 用 pred_dir 信心調整模型影響力 w（更像現實）
    # -----------------------------
    if pred_dir_last is None:
        conf = 0.35  # 不知道信心 → 保守
    else:
        try:
            pdv = float(pred_dir_last)
            conf = abs(pdv - 0.5) * 2.0  # 0~1
            conf = float(np.clip(conf, 0.0, 1.0))
        except Exception:
            conf = 0.35

    prices = [float(last_close)]
    for m in range(1, MONTHS + 1):
        phase_p = 2 * np.pi * (m * DPM) / float(cycle_p)
        phase_v = 2 * np.pi * (m * DPM) / float(cycle_v)

        cycle_main = base_amp * np.sin(phase_p)
        cycle_pull = 0.6 * base_amp * np.sin(phase_v + np.pi)

        # 月份越遠越不信模型；conf 越低也越不信
        w_time = float(np.exp(-0.55 * (m - 1)))
        w_conf = 0.25 + 0.75 * conf
        w = float(np.clip(w_time * w_conf, 0.05, 0.90))

        center = w * model_1m_price + (1 - w) * float(trend[m - 1])
        price = center * (1 + cycle_main + cycle_pull)
        prices.append(float(price))

    prices = np.array(prices, dtype=float)

    # -----------------------------
    # 8) ✅ Expected Range：用你自己 backtest 誤差校準（更貼近真實）
    # -----------------------------
    def load_recent_price_errors(ticker, max_files=90):
        files = sorted(glob.glob(f"results/*_{ticker}_backtest.csv"))[-max_files:]
        errs = []
        for f in files:
            try:
                bt = pd.read_csv(f)
                # 誤差：actual - pred（價格差）
                e = float(bt["actual_t1"].iloc[0]) - float(bt["pred_t1"].iloc[0])
                if np.isfinite(e):
                    errs.append(e)
            except Exception:
                pass
        return np.array(errs, dtype=float)

    errs = load_recent_price_errors(ticker)
    t = np.arange(len(prices), dtype=float)
    scale_t = np.sqrt(np.maximum(t, 1.0))  # √t 擴散

    if len(errs) >= 20:
        q10, q90 = np.quantile(errs, [0.10, 0.90])
        # 用回測誤差來擴散 band（價格差）
        upper = prices + float(q90) * scale_t
        lower = prices + float(q10) * scale_t
    else:
        # fallback：用 ATR 做 band（較粗，但不會亂）
        upper = prices * (1 + base_amp * (0.6 + 0.7 * (scale_t / scale_t.max())))
        lower = prices * (1 - base_amp * (0.6 + 0.7 * (scale_t / scale_t.max())))

    # -----------------------------
    # 9) X label
    # -----------------------------
    labels = ["Now"] + [
        (asof_date + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, MONTHS + 1)
    ]

    # -----------------------------
    # 10) Plot
    # -----------------------------
    plt.figure(figsize=(15, 7))
    x = np.arange(MONTHS + 1)

    plt.fill_between(x, lower, upper, alpha=0.18, label="Expected Range")
    plt.plot(x, prices, "r-o", linewidth=2.8, label="Projected Path")
    plt.scatter(0, prices[0], s=180, marker="*", label="Today")

    for i, p in enumerate(prices[1:]):
        plt.text(i + 1, p, f"{p:.2f}", ha="center", fontsize=12)

    info = (
        f"asof={asof_date.date()} | model_1M={model_1m_price:.2f} | amp={amp:.2f} | conf={conf:.2f}\n"
        f"drift(d)={daily_drift_adj:.5f} | trend_score={trend_score:.2f} | ATR%={atr_ratio:.2%} | RSI={float(rsi):.2f}\n"
        f"cycle_p={cycle_p} | cycle_v={cycle_v} | base_amp={base_amp:.3f} | edge(d)={edge_daily:.4f}"
    )
    plt.gca().text(
        0.01, 0.02, info,
        transform=plt.gca().transAxes,
        fontsize=9,
        alpha=0.55,
        ha="left",
        va="bottom"
    )

    plt.xticks(x, labels, fontsize=13)
    plt.title(f"{ticker} · 6M Outlook (Realistic 8110: drift+edge, calibrated band)")
    plt.grid(alpha=0.3)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    out = f"results/{datetime.now():%Y-%m-%d}_{ticker}_6m_advanced.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def last_valid_value(df: pd.DataFrame, col: str, lookback: int = 30):
    """
    取最近一筆有效（非 NaN）的指標值
    - 用於非交易日 / 補 today row 的情況
    """
    if col not in df.columns:
        return None

    s = df[col].iloc[-lookback:]
    s = s[s.notna()]
    if s.empty:
        return None
    return float(s.iloc[-1])



# ================= Main =================
# ================= Main =================
if __name__ == "__main__":
    TICKER = "8110.TW"
    COLLECTION = "NEW_stock_data_liteon"

    # ✅ 華東專屬設定（normalized return 版本）
    STOCK_CONFIG = {
        "8110.TW": {
            "LOOKBACK": 40,
            "STEPS": 5,                 # 5日：return head 用
            "MAX_DAILY_NORMRET": 3.0,   # normalized return 限幅（2~4 常見）
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

    # ✅ 趨勢 head 設定（最有感）
    TREND_H = 20      # 20 交易日 ≈ 1 個月趨勢
    K_FLAT  = 0.8     # 盤整門檻（0.6~1.2；越大越保守）

    os.makedirs("models", exist_ok=True)
    MODEL_PATH = f"models/{TICKER}_attn_lstm.keras"

    # ---------- Data ----------
    df = load_df_from_firestore(TICKER, collection=COLLECTION, days=500)
    df = ensure_latest_trading_row(df)
    df = add_features(df)

    # ✅ 華東專屬特徵（含 OHLC + 波動/跳空/量能）
    FEATURES = [
        "Close", "Open", "High", "Low",
        "Volume", "RSI", "MACD", "K", "D", "ATR_14",
        "HL_RANGE", "GAP", "VOL_REL",
        "TREND_60",
        "TREND_SLOPE_20"
    ]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"⚠️ Firestore 資料缺欄位：{missing}\n"
            f"請確認 catch_stock.py 寫回 8110.TW 時包含 Open/High/Low/Close/Volume，且指標欄位已寫入。"
        )

    if "RET_STD_20" not in df.columns:
        raise ValueError("⚠️ 缺少 RET_STD_20，請確認 add_features() 有被呼叫")

    df = df.dropna()

    # ✅ create_sequences：會回傳 y_trend3（3類趨勢）
    X, y_ret, y_dir, y_trend3, idx = create_sequences(
        df, FEATURES,
        steps=STEPS,
        window=LOOKBACK,
        trend_h=TREND_H,
        k_flat=K_FLAT
    )
    print(f"df rows: {len(df)} | X samples: {len(X)}")

    if len(X) < 60:
        raise ValueError("⚠️ 可用序列太少（<60）。建議：降低 LOOKBACK 或增加 days/檢查 NaN。")

    # ---------- Time-series split ----------
    split = int(len(X) * 0.85)
    X_tr, X_va = X[:split], X[split:]
    y_ret_tr, y_ret_va = y_ret[:split], y_ret[split:]
    y_dir_tr, y_dir_va = y_dir[:split], y_dir[split:]
    y_tr3_tr, y_tr3_va = y_trend3[:split], y_trend3[split:]
    idx_tr, idx_va = idx[:split], idx[split:]

    # ✅ scaler.fit 僅用 train 區間（避免 leakage）
    train_end_date = pd.Timestamp(idx_tr[-1])
    df_for_scaler = df.loc[:train_end_date, FEATURES].copy()

    if len(df_for_scaler) < LOOKBACK + max(STEPS, TREND_H) + 5:
        raise ValueError("⚠️ train 區間太短，無法穩定 fit scaler。請確認資料量或調整 LOOKBACK。")

    sx = MinMaxScaler()
    sx.fit(df_for_scaler.values)

    def scale_X(Xb):
        n, t, f = Xb.shape
        return sx.transform(Xb.reshape(-1, f)).reshape(n, t, f)

    X_tr_s = scale_X(X_tr)
    X_va_s = scale_X(X_va)

    # ---------- Model ----------
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

    # ✅ 真正時間序列 validation（最後 15%）
    model.fit(
        X_tr_s,
        {"return": y_ret_tr, "direction": y_dir_tr, "trend3": y_tr3_tr},
        validation_data=(X_va_s, {"return": y_ret_va, "direction": y_dir_va, "trend3": y_tr3_va}),
        epochs=120,
        batch_size=16,
        verbose=2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)]
    )

    model.save(MODEL_PATH)
    print(f"💾 已儲存模型：{MODEL_PATH}")

    # ---------- Predict (use validation tail as "latest unseen") ----------
    pred_ret, pred_dir, pred_tr3 = model.predict(X_va_s, verbose=0)

    raw_norm_returns = pred_ret[-1]         # 5日 normalized return（已限幅）
    p_dir = float(pred_dir[-1][0])          # 5日看漲機率
    p_tr = pred_tr3[-1].astype(float)       # 20日趨勢三類 [Down, Flat, Up]
    trend_label = ["Down", "Flat", "Up"][int(np.argmax(p_tr))]

    print(f"📈 5D 看漲機率: {p_dir:.2%}")
    print(f"📌 20D 趨勢: {trend_label} | P(Down/Flat/Up) = {p_tr[0]:.2f}/{p_tr[1]:.2f}/{p_tr[2]:.2f}")

    # ---------- Asof date ----------
    asof_date, is_today_trading = get_asof_trading_day(df)
    if not is_today_trading:
        print(f"ℹ️ 今日非交易日，{TICKER} 使用最近交易日 {asof_date.date()}")

    last_close = float(df.loc[asof_date, "Close"])

    # ✅ 把 normalized return 乘回波動尺度（用 asof 的 RET_STD_20）
    scale_last = float(df.loc[asof_date, "RET_STD_20"])
    if not np.isfinite(scale_last) or scale_last <= 0:
        scale_last = float(np.log(df["Close"].astype(float)).diff().rolling(20).std().iloc[-1])
    scale_last = max(scale_last, 1e-6)

    # 🔧 Regime-based amp（保留給 6M 週期震盪用）
    trend60 = last_valid_value(df, "TREND_60", lookback=5)
    amp = 1.0
    if trend60 is not None:
        if trend60 > 1.0:
            amp = 1.4
        elif trend60 < -1.0:
            amp = 1.3
        elif abs(trend60) < 0.5:
            amp = 0.6

    print(f"📊 Regime amp = {amp:.2f}")

    # ---------- 5D price projection ----------
    # ✅ 最有感修正：5日推回價格不要乘 amp（避免放大噪音）
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
        start=asof_date + BDay(1),
        periods=STEPS
    )

    # ✅ 預測數值輸出 CSV（檔名含 ticker）
    os.makedirs("results", exist_ok=True)
    forecast_csv = f"results/{datetime.now():%Y-%m-%d}_{TICKER}_forecast.csv"
    future_df.to_csv(forecast_csv, index=False, encoding="utf-8-sig")

    # ✅ 圖輸出（內容不動、檔名含 ticker）
    plot_and_save(df, future_df, ticker=TICKER)
    plot_backtest_error(df, ticker=TICKER)

    # ---------- 6M Outlook (advanced) ----------
    # 用最後一筆方向機率做 conf（你原本設計）
    pred_dir_last = float(p_dir)

    plot_6m_trend_advanced(
        df=df,
        last_close=last_close,
        raw_norm_returns=raw_norm_returns,
        scale_last=scale_last,
        ticker=TICKER,
        asof_date=asof_date,
        amp=amp,
        pred_ret_all=pred_ret,          # ✅ 可用 ensemble
        pred_dir_last=pred_dir_last,
        k_ens=20
    )
