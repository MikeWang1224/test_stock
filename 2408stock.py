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



# ================= Feature Engineering =================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ===== Volume 穩定化 =====
    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].astype(float))

    close = df["Close"].astype(float)

    # ===== log return =====
    logret = np.log(close).diff()

    # ===== RET_STD_20（給 normalized return 用）=====
    df["RET_STD_20"] = logret.rolling(20).std()

    # ===== 圖表用均線（不影響模型）=====
    df["SMA5"] = close.rolling(5).mean()
    df["SMA10"] = close.rolling(10).mean()

    return df


# ================= Sequence（避免錯位，且不亂切 df） =================
def create_sequences(df, features, steps=5, window=40):
    """
    X: t-window ~ t-1
    y_ret: t ~ t+steps-1 的 log return
    y_dir: 未來 steps 天累積方向（sum future_ret > 0）
    idx: 每個樣本對應的「t 當天日期」
    """
    X, y_ret, y_dir, idx = [], [], [], []

    close = df["Close"].astype(float)
    logret = np.log(close).diff()
    ret_std = df["RET_STD_20"].astype(float).values
    feat = df[features].values
    
    for i in range(window, len(df) - steps):
        x_seq = feat[i - window:i]
    
        scale = ret_std[i - 1]   # 用 t-1 的波動
        if not np.isfinite(scale) or scale <= 0:
            continue
    
        future_ret = logret.iloc[i:i + steps].values / scale
    
        if np.any(np.isnan(future_ret)) or np.any(np.isnan(x_seq)):
            continue
    
        X.append(x_seq)
        y_ret.append(future_ret)
        y_dir.append(1.0 if future_ret.sum() > 0 else 0.0)
        idx.append(df.index[i])


    return np.array(X), np.array(y_ret), np.array(y_dir), np.array(idx)

# ================= Loss（direction 用 focal；不支援就 fallback） =================
def get_direction_loss():
    if hasattr(tf.keras.losses, "BinaryFocalCrossentropy"):
        return tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)

    def weighted_bce(y_true, y_pred, pos_weight=1.5):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        w = y_true * pos_weight + (1.0 - y_true) * 1.0
        return tf.reduce_mean(w * bce)

    return weighted_bce

# ================= Model build（return 限幅 + 方向與return對齊） =================
def build_attention_lstm(input_shape, steps, max_daily_logret=0.06, dir_from_ret_weight=2.0):
    inp = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)

    score = Dense(1, name="attn_score")(x)
    weights = Softmax(axis=1, name="attn_weights")(score)
    context = Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1),
                     name="attn_context")([x, weights])

    raw = Dense(steps, activation="tanh", name="raw_returns")(context)
    out_ret = Lambda(lambda t: t * max_daily_logret, name="return")(raw)

    base_logit = Dense(1, activation=None, name="dir_base_logit")(context)
    sum_raw = Lambda(lambda r: tf.reduce_sum(r, axis=1, keepdims=True), name="sum_raw")(raw)
    dir_logit = Lambda(lambda t: t[0] + dir_from_ret_weight * t[1], name="dir_logit")([base_logit, sum_raw])
    out_dir = Lambda(lambda z: tf.sigmoid(z), name="direction")(dir_logit)

    model = Model(inp, [out_ret, out_dir])
    return model

def compile_model(model, direction_weight=0.8, lr=7e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "return": tf.keras.losses.Huber(),
            "direction": get_direction_loss()
        },
        loss_weights={
            "return": 1.0,
            "direction": float(direction_weight)
        },
        metrics={
            "direction": [
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc")
            ]
        }
    )
    return model

# ================= 原預測圖（Today 標記，檔名加 ticker） =================
def plot_and_save(df_hist, future_df, ticker):
    hist = df_hist.tail(10)
    hist_dates = hist.index.strftime("%m-%d").tolist()
    future_dates = future_df["date"].dt.strftime("%m-%d").tolist()

    all_dates = hist_dates + future_dates
    x_hist = np.arange(len(hist_dates))
    x_future = np.arange(len(hist_dates), len(all_dates))

    plt.figure(figsize=(18, 8))
    ax = plt.gca()

    ax.plot(x_hist, hist["Close"], label="Close")
    ax.plot(x_hist, hist["SMA5"], label="SMA5")
    ax.plot(x_hist, hist["SMA10"], label="SMA10")

    today_x = x_hist[-1]
    today_y = float(hist["Close"].iloc[-1])
    ax.scatter([today_x], [today_y], marker="*", s=160, label="Today Close")
    ax.text(today_x, today_y + 0.3, f"Today {today_y:.2f}", fontsize=17, ha="center")

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["Close"].iloc[-1]] + future_df["Pred_Close"].tolist(),
        "r:o", label="Pred Close"
    )

    for i, price in enumerate(future_df["Pred_Close"]):
        ax.text(x_future[i], price + 0.3, f"{price:.2f}", color="red", fontsize=15, ha="center")

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
    ax.set_title(f"{ticker} Attention-LSTM 預測")

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{datetime.now():%Y-%m-%d}_{ticker}_pred.png", dpi=300, bbox_inches="tight")
    plt.close()

# ================= 回測決策分岔圖（PNG + CSV） =================
# ================= 回測決策分岔圖（PNG + CSV） =================
def plot_backtest_error(df, ticker: str):
    """
    決策式回測圖（Decision-based Backtest）

    嚴格定義：
    - 回測一定使用「昨天或更早」產生的 forecast
    - t / t+1 為最後兩個『已完成的真實交易日』
    - 絕不使用今天 forecast（避免偷看未來）
    """

    # === 1️⃣ 只保留「真實交易日」（排除 ensure_latest_trading_row 補的假日）===
    real_df = df.copy()
    real_df = real_df[real_df["Close"].diff().abs() > 1e-9]

    if len(real_df) < 3:
        print("⚠️ 真實交易日不足，略過回測")
        return

    # === 2️⃣ 定義 t / t+1（最後兩個完成交易日）===
    valid_days = real_df.index
    t  = valid_days[-2]   # decision day（昨天）
    t1 = valid_days[-1]   # actual day（今天已收盤）

    # === 3️⃣ 從 results 找「≤ t 的最近一筆 forecast」===
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

        # ⭐ 核心條件：forecast 日期 ≤ 決策日 t
        if d <= t:
            forecast_files.append((d, f))

    if not forecast_files:
        print("⚠️ 找不到 ≤ t 的歷史 forecast，略過回測")
        return

    forecast_date = asof_date.normalize()

    forecast_name = f"{forecast_date:%Y-%m-%d}_{ticker}_forecast.csv"
    forecast_csv = os.path.join("results", forecast_name)
    
    if not os.path.exists(forecast_csv):
        print(f"⚠️ 找不到 {forecast_name}，略過回測")
        return

    forecast_csv = os.path.join("results", forecast_name)

    print(f"📄 Backtest 使用 forecast：{forecast_name}")

    future_df = pd.read_csv(forecast_csv, parse_dates=["date"])

    # === 4️⃣ 取數值（完全對齊交易語意）===
    close_t   = float(real_df.loc[t, "Close"])
    actual_t1 = float(real_df.loc[t1, "Close"])
    pred_t1   = float(future_df.loc[0, "Pred_Close"])

    # === 5️⃣ 畫圖資料「只用真實交易日」===
    trend = real_df.loc[:t].tail(4)
    x_trend = np.arange(len(trend))
    x_t = x_trend[-1]

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    ax.plot(x_trend, trend["Close"], "k-o", label="Recent Close")
    ax.plot([x_t, x_t + 1], [close_t, pred_t1],
            "r--o", linewidth=2.5, label="Pred (t → t+1)")
    ax.plot([x_t, x_t + 1], [close_t, actual_t1],
            "g-o", linewidth=2.5, label="Actual (t → t+1)")

    dx = 0.08
    price_offset = max(0.2, close_t * 0.002)

    ax.text(x_t, close_t + price_offset,
            f"{close_t:.2f}", ha="center", va="bottom", fontsize=18)
    ax.text(x_t + 1 + dx, pred_t1,
            f"Pred {pred_t1:.2f}", ha="left", va="center",
            fontsize=16, color="red")
    ax.text(x_t + 1 + dx, actual_t1,
            f"Actual {actual_t1:.2f}", ha="left", va="center",
            fontsize=16, color="green")

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
        fontsize=8,
        alpha=0.4,
        ha="left",
        va="bottom"
    )

    os.makedirs("results", exist_ok=True)
    out_png = f"results/{t1:%Y-%m-%d}_{ticker}_backtest.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    # === 6️⃣ CSV 輸出 ===
    bt = pd.DataFrame([{
        "forecast_date": forecast_date.date(),
        "decision_day": t.date(),
        "actual_day": t1.date(),
        "close_t": close_t,
        "pred_t1": pred_t1,
        "actual_t1": actual_t1,
        "direction_pred": int(np.sign(pred_t1 - close_t)),
        "direction_actual": int(np.sign(actual_t1 - close_t))
    }])

    out_csv = f"results/{t1:%Y-%m-%d}_{ticker}_backtest.csv"
    bt.to_csv(out_csv, index=False, encoding="utf-8-sig")

# ================= 6M Trend Plot（x 軸 = 月） =================
def plot_6m_trend_advanced(
    df: pd.DataFrame,
    last_close: float,
    raw_norm_returns: np.ndarray,
    scale_last: float,
    ticker: str,
    asof_date: pd.Timestamp
):
    MONTHS = 6
    DPM = 21

    # =============================
    # 1️⃣ 主升趨勢（模型）
    # =============================
    # =============================
# 1️⃣ 主升趨勢（低頻，來自歷史價格）
# =============================
# 用近 120 個交易日估計「長期 drift」
    log_price = np.log(df["Close"].astype(float))
    ret_ewm = log_price.diff().ewm(span=60).mean()
    
    daily_drift = float(ret_ewm.iloc[-1])
    daily_drift = np.clip(daily_drift, -0.01, 0.01)  # 防爆（±1% / day）


    # ===== Regime 判斷（Priority 1）=====
    atr = last_valid_value(df, "ATR_14", lookback=40)
    rsi = last_valid_value(df, "RSI", lookback=40)
    
    # 波動強度（相對價格）
    vol_regime = atr / last_close if atr else 0.03
    
    # 趨勢可信度分數（0~1）
    trend_score = 1.0
    
    # 1️⃣ 高檔過熱 → drift 不可信
    if rsi and rsi > 75:
        trend_score *= 0.3
    elif rsi and rsi > 65:
        trend_score *= 0.6
    
    # 2️⃣ 超低波動 → 偏盤整
    if vol_regime < 0.015:
        trend_score *= 0.5
    
    # 3️⃣ 超高波動 → regime 不穩
    if vol_regime > 0.08:
        trend_score *= 0.7
    
    # 最終調整 drift
    daily_drift *= trend_score

      
    monthly_logret = daily_drift * DPM
    
    trend = []
    p = last_close
    for _ in range(MONTHS):
        p *= np.exp(monthly_logret)
        trend.append(p)
    
    trend = np.array(trend)
    


    # =============================
    # 2️⃣ 主週期（價格）
    # =============================
    close = df["Close"].iloc[-180:].values
    close = close - close.mean()

    fft_p = np.fft.rfft(close)
    freq_p = np.fft.rfftfreq(len(close), d=1)
    idx_p = np.argmax(np.abs(fft_p[1:])) + 1
    cycle_p = np.clip(int(round(1 / freq_p[idx_p])), 40, 120)

    # =============================
# 3️⃣ 回檔週期（成交量）
# =============================
    vol_series = df["Volume"].iloc[-180:].dropna().values
    
    if len(vol_series) < 60:
        cycle_v = 30  # fallback
    else:
        vol_centered = vol_series - vol_series.mean()
    
        fft_v = np.fft.rfft(vol_centered)
        freq_v = np.fft.rfftfreq(len(vol_centered), d=1)
        idx_v = np.argmax(np.abs(fft_v[1:])) + 1
        cycle_v = np.clip(int(round(1 / freq_v[idx_v])), 20, 60)


    # =============================
    # 4️⃣ 震盪幅度（ATR × RSI）
    # =============================
    atr = last_valid_value(df, "ATR_14", lookback=40)
    if atr is None:
        raise ValueError("❌ 無可用 ATR_14（最近 40 日皆為 NaN）")
    atr_ratio = atr / last_close

    rsi = last_valid_value(df, "RSI", lookback=40)
    rsi_factor = np.clip(abs(rsi - 50) / 50, 0.3, 1.2)

    base_amp = atr_ratio * rsi_factor
    base_amp = np.clip(base_amp, 0.02, 0.18)

    # =============================
    # 5️⃣ 合成價格（多週期）
    # =============================
    prices = [last_close]

    for m in range(1, MONTHS + 1):
        phase_p = 2 * np.pi * (m * DPM) / cycle_p
        phase_v = 2 * np.pi * (m * DPM) / cycle_v

        cycle_main = base_amp * np.sin(phase_p)
        cycle_pull = 0.6 * base_amp * np.sin(phase_v + np.pi)

        price = trend[m - 1] * (1 + cycle_main + cycle_pull)
        prices.append(price)

    prices = np.array(prices)

    # =============================
    # 6️⃣ 區間帶（ATR-based fan）
    # =============================
    time_scale = np.linspace(0.6, 1.3, len(prices))
    upper = prices * (1 + base_amp * time_scale)
    lower = prices * (1 - base_amp * time_scale)


    # =============================
    # 7️⃣ X 軸（月）
    # =============================
    labels = ["Now"] + pd.date_range(
        asof_date + pd.offsets.MonthBegin(1),
        periods=MONTHS,
        freq="MS"
    ).strftime("%Y-%m").tolist()

    # =============================
    # 8️⃣ Plot
    # =============================
    plt.figure(figsize=(15, 7))
    x = np.arange(MONTHS + 1)

    plt.fill_between(x, lower, upper, alpha=0.18, label="Expected Range")
    plt.plot(x, prices, "r-o", linewidth=2.8, label="Projected Path")
    plt.scatter(0, prices[0], s=180, marker="*", label="Today")

    for i, p in enumerate(prices[1:]):
        plt.text(i + 1, p, f"{p:.2f}", ha="center", fontsize=12)

    plt.xticks(x, labels, fontsize=13)
    plt.title(f"{ticker} · 6M Outlook (Multi-Cycle + ATR + RSI)")
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
if __name__ == "__main__":
    TICKER = "2408.TW"
    COLLECTION = "NEW_stock_data_liteon"

    # ✅ 華東專屬設定（normalized return 版本）
    STOCK_CONFIG = {
        "2408.TW": {
            "LOOKBACK": 40,
            "STEPS": 5,
            "MAX_DAILY_NORMRET": 3.0,
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
    df = ensure_latest_trading_row(df)
    df = add_features(df)

    FEATURES = [
        "Close",
        "Volume",
        "RSI",
        "MACD",
        "K",
        "D",
        "ATR_14"
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
            max_daily_logret=cfg["MAX_DAILY_NORMRET"]
        )
        model = compile_model(
          model,
          direction_weight=0.8,
          lr=cfg["LR"]
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

    asof_date, is_today_trading = get_asof_trading_day(df)

    if not is_today_trading:
        print(f"ℹ️ 今日非交易日，8110.TW 使用最近交易日 {asof_date.date()}")
    
    last_close = float(df.loc[asof_date, "Close"])


    # ✅ 把 normalized return 乘回波動尺度（用 asof 的 RET_STD_20）
    scale_last = float(df.loc[asof_date, "RET_STD_20"])
    if not np.isfinite(scale_last) or scale_last <= 0:
        # fallback：用最近 20 天 std 估
        scale_last = float(np.log(df["Close"].astype(float)).diff().rolling(20).std().iloc[-1])
    scale_last = max(scale_last, 1e-6)


    # 🔧 ADD: Regime-based 波段放大 / 壓縮（用最近的 TREND_60）
    trend60 = last_valid_value(df, "TREND_60", lookback=5)
    
    amp = 1.0
    if trend60 is not None:
        if trend60 > 1.0:
            amp = 1.2
        elif trend60 < -1.0:
            amp = 1.1
        else:
            amp = 0.8    # 盤整 → 壓縮
    
    print(f"📊 Regime amp = {amp:.2f}")

    prices = []
    price = last_close
    for r_norm in raw_norm_returns:
        r = float(r_norm) * scale_last * amp
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
    forecast_csv = f"results/{asof_date:%Y-%m-%d}_{TICKER}_forecast.csv"
    future_df.to_csv(forecast_csv, index=False, encoding="utf-8-sig")

    # ✅ 圖輸出（內容不動、檔名改含 ticker）
    plot_and_save(df, future_df, ticker=TICKER)
    plot_backtest_error(df, ticker=TICKER)
    # ================= 6M Trend Forecast（x 軸 = 月） =================
    plot_6m_trend_advanced(
        df=df,
        last_close=last_close,
        raw_norm_returns=raw_norm_returns,
        scale_last=scale_last,
        ticker=TICKER,
        asof_date=asof_date
    )
