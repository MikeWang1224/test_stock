# -*- coding: utf-8 -*-
"""
FireBase_Attention_LSTM_Direction.py (2408.TW 南亞科｜方向更準版 + 更穩版)

你要的「模型端」重點改動（最少但最有感）：
1) ✅ 加入時間序 validation（EarlyStopping 監看 val_loss，不再假穩）
2) ✅ direction 改用 Focal loss（或 TF 不支援時 fallback 成加權 BCE）
3) ✅ direction head 與 return head 對齊：把「sum(raw_returns)」加到方向 logit（避免一個說漲一個說跌）
4) ✅ scaler 存檔/載入（續訓不再每天換座標系）
5) ✅ cap 寫入 meta.json：續訓時沿用同一個 cap（避免模型圖裡 cap 固定卻以為更新了）

✅ NEW：把 Firestore 的外生因子加入模型（不改 Firestore 任何資料位置）
- TAIEX / ELECTRONICS / USD_TWD：同日對齊
- SOX / MU_US：以「美股收盤 -> 台股下一個交易日」方式對齊（index + BDay(1)）

⚠️ 圖表與輸出檔名規則不變（你的 results/xxxx 檔案格式維持原樣）
"""

import os, json, random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
import joblib  # ✅ scaler persistence

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Softmax, Lambda
from tensorflow.keras.callbacks import EarlyStopping

from zoneinfo import ZoneInfo
now_tw = datetime.now(ZoneInfo("Asia/Taipei"))

# ================= Seed（讓結果更穩、可比較） =================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
else:
    print("⚠️ FIREBASE 未設定，Firestore 讀取將無資料")

# ================= Firestore 讀取（個股） =================
def load_df_from_firestore(ticker, collection="NEW_stock_data_liteon", days=500):
    rows = []
    if db:
        for doc in db.collection(collection).stream():
            p = doc.to_dict().get(ticker)
            if p:
                rows.append({"date": doc.id, **p})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"⚠️ Firestore 無資料：{ticker}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days).set_index("date")
    return df

# ================= Firestore 讀取（外生因子 Close only） =================
def load_factor_close_from_firestore(alias, collection="NEW_stock_data_liteon", days=800):
    """
    讀取 Firestore 文件中的 {alias: {Close: ...}}，回傳 Series(index=date, value=Close)
    alias 例：TAIEX / ELECTRONICS / USD_TWD / SOX / MU_US
    """
    rows = []
    if db:
        for doc in db.collection(collection).stream():
            p = doc.to_dict().get(alias)
            if isinstance(p, dict) and "Close" in p:
                rows.append({"date": doc.id, "Close": p["Close"]})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"⚠️ Firestore 無資料：{alias}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days).set_index("date")
    s = df["Close"].astype(float)
    s.name = alias
    return s

def attach_factors_to_stock_df(df_stock, collection="NEW_stock_data_liteon"):
    """
    df_stock: 2408 的 df（index=台股交易日）
    - 台股/匯率因子（TAIEX/ELECTRONICS/USD_TWD）：直接 reindex + ffill + bfill
    - 美股因子（SOX/MU_US）：把美股日期往後推 1 個 BDay，落在台股下一交易日，再 reindex + ffill + bfill
    ⚠️ 只改 DataFrame（記憶體內），不會改 Firestore 任何資料。
    """
    df_stock = df_stock.copy()
    idx = df_stock.index

    # 台股/匯率：同日對齊
    for a in ["TAIEX", "ELECTRONICS", "USD_TWD"]:
        try:
            s = load_factor_close_from_firestore(a, collection=collection)
            # ✅ 重要：ffill + bfill，避免一開始一串 NaN 直接把整段砍掉
            df_stock[a] = s.reindex(idx).ffill().bfill()
        except Exception as e:
            print(f"⚠️ 無法載入 {a}: {e}")
            df_stock[a] = np.nan

    # 美股：美股 D 的 Close -> 台股 D+1
    for a in ["SOX", "MU_US"]:
        try:
            s = load_factor_close_from_firestore(a, collection=collection)
            s_shifted = s.copy()
            s_shifted.index = (s_shifted.index + BDay(1))
            s_shifted.name = a
            # ✅ 同樣補齊
            df_stock[a] = s_shifted.reindex(idx).ffill().bfill()
        except Exception as e:
            print(f"⚠️ 無法載入 {a}: {e}")
            df_stock[a] = np.nan

    return df_stock

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
    # ✅ Volume 尺度穩定
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
    y_dir: 未來 steps 天累積方向（sum future_ret > 0）
    idx: 每個樣本對應的「t 當天日期」
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

# ================= 回測決策分岔圖（PNG + CSV，讀對應 ticker forecast） =================
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

    forecast_files.sort(key=lambda x: x[0], reverse=True)
    forecast_date, forecast_name = forecast_files[0]
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
    TICKER = "2408.TW"
    LOOKBACK = 40
    STEPS = 5
    COLLECTION = "NEW_stock_data_liteon"

    os.makedirs("results", exist_ok=True)
    MODEL_PATH  = f"results/{TICKER}_model.keras"
    SCALER_PATH = f"results/{TICKER}_scaler.pkl"
    META_PATH   = f"results/{TICKER}_meta.json"

    df = load_df_from_firestore(TICKER, collection=COLLECTION, days=500)
    #df = ensure_today_row(df)
    df = add_features(df)

    # ✅ NEW：接外生因子（只改 DataFrame，不改 Firestore）
    df = attach_factors_to_stock_df(df, collection=COLLECTION)

    FEATURES = [
        "Close", "Volume", "RSI", "MACD", "K", "D", "ATR_14",
        "TAIEX", "ELECTRONICS", "USD_TWD", "SOX", "MU_US"
    ]

    cols_check = [c for c in ["Close", "TAIEX", "ELECTRONICS", "USD_TWD", "SOX", "MU_US"] if c in df.columns]
    print("🔎 factors tail:\n", df[cols_check].tail(5))

    # ✅ 關鍵修正 1：不要整張 df.dropna()，只針對模型 FEATURES
    df = df.dropna(subset=FEATURES)

    X, y_ret, y_dir, idx = create_sequences(df, FEATURES, steps=STEPS, window=LOOKBACK)
    print(f"{TICKER} | df rows: {len(df)} | X samples: {len(X)}")

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

    if os.path.exists(SCALER_PATH):
        sx = joblib.load(SCALER_PATH)
        print(f"✅ Load scaler: {SCALER_PATH}")
    else:
        sx = MinMaxScaler()
        sx.fit(df_for_scaler.values)
        joblib.dump(sx, SCALER_PATH)
        print(f"💾 Scaler saved: {SCALER_PATH}")

    def scale_X(Xb):
        n, t, f = Xb.shape
        return sx.transform(Xb.reshape(-1, f)).reshape(n, t, f)

    X_tr_s = scale_X(X_tr)
    X_te_s = scale_X(X_te)

    train_close = df.loc[:train_end_date, "Close"].astype(float)
    train_logret_abs = np.log(train_close).diff().dropna().abs()

    auto_cap = float(train_logret_abs.quantile(0.99))
    auto_cap = float(np.clip(auto_cap, 0.03, 0.10))
    print(f"✅ max_daily_logret auto (99% quantile, clipped): {auto_cap:.4f}")

    meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    if "cap" in meta:
        cap_used = float(meta["cap"])
        if abs(cap_used - auto_cap) > 1e-6:
            print(f"⚠️ cap 已固定沿用 meta cap={cap_used:.4f}（auto_cap={auto_cap:.4f} 不套用）")
    else:
        cap_used = auto_cap
        meta = {
            "ticker": TICKER,
            "lookback": LOOKBACK,
            "steps": STEPS,
            "features": FEATURES,
            "cap": cap_used,
            "created_at_tw": f"{now_tw:%Y-%m-%d %H:%M:%S}"
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"💾 Meta saved: {META_PATH} (cap={cap_used:.4f})")

    DIRECTION_WEIGHT = 0.8

    n_tr = len(X_tr_s)
    val_cut = int(n_tr * 0.90)
    if val_cut < 10:
        raise ValueError("⚠️ train 太少，無法切 val。請增加資料或降低 LOOKBACK/STEPS。")

    X_fit, X_val = X_tr_s[:val_cut], X_tr_s[val_cut:]
    y_ret_fit, y_ret_val = y_ret_tr[:val_cut], y_ret_tr[val_cut:]
    y_dir_fit, y_dir_val = y_dir_tr[:val_cut], y_dir_tr[val_cut:]

    print(f"✅ Fit samples: {len(X_fit)} | Val samples: {len(X_val)}")

    if os.path.exists(MODEL_PATH):
        print(f"✅ Load existing model: {MODEL_PATH}")
        model = load_model(MODEL_PATH, safe_mode=False)
        model = compile_model(model, direction_weight=DIRECTION_WEIGHT, lr=3e-4)
    else:
        print("✅ Build new model")
        model = build_attention_lstm(
            (LOOKBACK, len(FEATURES)),
            STEPS,
            max_daily_logret=cap_used,
            dir_from_ret_weight=2.0
        )
        model = compile_model(model, direction_weight=DIRECTION_WEIGHT, lr=7e-4)

    model.fit(
        X_fit,
        {"return": y_ret_fit, "direction": y_dir_fit},
        validation_data=(X_val, {"return": y_ret_val, "direction": y_dir_val}),
        epochs=80,
        batch_size=16,
        verbose=2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
    )

    model.save(MODEL_PATH)
    print(f"💾 Model saved: {MODEL_PATH}")

    pred_ret, pred_dir = model.predict(X_te_s, verbose=0)
    raw_returns = pred_ret[-1]

    print(f"📈 {TICKER} 預測方向機率（看漲）: {pred_dir[-1][0]:.2%}")

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
    last_trade_date = df.index[-1]

    # ================= 生成未來交易日（台股實際交易日） =================
    # 從 df index 找到 asof_date 的位置
    asof_idx = df.index.get_loc(asof_date)
    future_dates = df.index[asof_idx + 1 : asof_idx + 1 + STEPS]
    
    # 若資料不足 STEPS 天，補最後一天（避免報錯）
    if len(future_dates) < STEPS:
        last_date = df.index[-1]
        while len(future_dates) < STEPS:
            future_dates = future_dates.append(pd.DatetimeIndex([last_date]))
    
    future_df["date"] = future_dates



    future_df.to_csv(
        f"results/{datetime.now():%Y-%m-%d}_{TICKER}_forecast.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_and_save(df, future_df, TICKER)
    plot_backtest_error(df, TICKER)
