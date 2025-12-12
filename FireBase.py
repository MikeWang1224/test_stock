# -*- coding: utf-8 -*-
"""
完整整合版：TCN + multi-step returns (預測未來 10 日 return) -> 反推 Close -> 計算 MA5/MA10
功能：
 - 下載歷史 (yfinance)
 - 計算技術指標（EMA、SMA、MACD hist、RSI、StochRSI、HV、ATR、BB、OBV、lag features、時間因子）
 - 建 dataset (X: 多特徵時間序列, y: 未來 10 天 return)
 - 時序 split (walk-forward-ish using time_series_split)
 - 建 TCN 模型訓練/驗證
 - 預測並反推未來 Close / MA
 - 繪圖、上傳 Firebase Storage（若設定）
 - 寫入 Firestore 歷史資料與預測（若設定）
"""
import os, json, math, random
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ML / DL
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Firebase (optional)
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage
from pandas.tseries.offsets import BDay

# ---------------- Firebase 初始化（含 Storage） ----------------
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None
bucket = None
storage_client = None

if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(cred, {"storageBucket": f"{key_dict.get('project_id')}.appspot.com"})
    db = firestore.client()
    try:
        storage_client = storage.Client.from_service_account_info(key_dict)
        bucket = storage_client.bucket(f"{key_dict.get('project_id')}.appspot.com")
    except Exception as e:
        print("⚠️ Storage client 初始化失敗，Storage 功能停用:", e)
        bucket = None
else:
    print("⚠️ FIREBASE env 未設定 — 會略過上傳與寫入步驟")

# ---------------- 指標 / 特徵函式 ----------------
def add_basic_indicators(df):
    df = df.copy()
    # SMA
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    # EMA
    for span in [5,10,20,50,100]:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    # MACD hist
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_hist'] = macd - signal
    # RSI (14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # Stochastic RSI (StochRSI) approximate
    rsi = df['RSI_14']
    df['StochRSI'] = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
    # ATR (14)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    # Bollinger
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_SMA_20'] = df['OBV'].rolling(20).mean()
    return df

def add_lag_time_features(df, lags=[1,2,3,5,10], rolling_windows=[5,10,20]):
    df = df.copy()
    for lag in lags:
        df[f'ret_lag_{lag}'] = df['Close'].pct_change(lag).fillna(0)
        df[f'logret_lag_{lag}'] = np.log(df['Close'] / df['Close'].shift(lag)).replace([np.inf, -np.inf], 0).fillna(0)
    for w in rolling_windows:
        df[f'vol_{w}'] = df['Close'].pct_change().rolling(window=w).std().fillna(0)
        df[f'meanret_{w}'] = df['Close'].pct_change().rolling(window=w).mean().fillna(0)
    # 時間因子
    df['day_of_week'] = df.index.dayofweek
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['month'] = df.index.month
    # boolean regime-like features
    df['close_above_sma5'] = (df['Close'] > df['SMA_5']).astype(int)
    df['macd_pos'] = (df['MACD'] > 0).astype(int)
    df = df.fillna(0)
    return df

# ---------------- fetch & prepare ----------------
def fetch_and_prepare(ticker="2301.TW", period="36mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError("yfinance returned empty dataframe")
    df = add_basic_indicators(df)
    df = add_lag_time_features(df)
    df = df.dropna()
    return df

# ---------------- 更新今天 Close 從 Firestore（若有） ----------------
def update_today_from_firestore(df):
    if db is None:
        return df
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        doc_ref = db.collection("NEW_stock_data_liteon").document(today_str)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict().get("2301.TW", {})
            if "Close" in data:
                try:
                    df.loc[pd.Timestamp(today_str), 'Close'] = float(data["Close"])
                except Exception:
                    pass
    except Exception:
        pass
    df = df.dropna()
    return df

# ---------------- Save historical to Firestore ----------------
def save_stock_data_to_firestore(df, ticker="2301.TW", collection_name="NEW_stock_data_liteon"):
    if db is None:
        print("⚠️ Firebase 未啟用，略過寫入股票資料")
        return
    batch = db.batch()
    count = 0
    try:
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            try:
                payload = {
                    "Close": float(row["Close"]),
                    "Volume": float(row["Volume"]),
                    "MACD": float(row.get("MACD", 0.0)),
                    "RSI": float(row.get("RSI_14", 0.0)),
                    "K": float(row.get("StochRSI", 0.0)),
                    "D": float(row.get("SMA_5", 0.0))  # placeholder if needed
                }
            except Exception:
                continue
            doc_ref = db.collection(collection_name).document(date_str)
            batch.set(doc_ref, {ticker: payload})
            count += 1
            if count >= 300:
                batch.commit()
                batch = db.batch()
                count = 0
        if count > 0:
            batch.commit()
        print(f"🔥 歷史股票資料已寫入 Firestore （collection: {collection_name}）")
    except Exception as e:
        print("❌ 寫入 Firestore 發生錯誤：", e)

# ---------------- Dataset builder: returns target ----------------
def build_multi_step_returns(df_features, close_col="Close", pred_steps=10, lookback=60, feature_cols=None):
    """
    df_features: dataframe containing all desired features (index = dates)
    Returns: X (n_samples, lookback, n_features), y (n_samples, pred_steps) as returns
    """
    if feature_cols is None:
        feature_cols = df_features.columns.tolist()
    closes = df_features[close_col].values
    # returns: (t+1)/t - 1
    returns = closes[1:] / closes[:-1] - 1
    returns = np.append(returns, 0)  # pad last
    X_list, y_list = [], []
    for i in range(len(df_features) - lookback - pred_steps):
        X_list.append(df_features[feature_cols].iloc[i:i+lookback].values)
        future_returns = returns[i+lookback : i+lookback+pred_steps]
        y_list.append(future_returns)
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, feature_cols

# ---------------- TCN 模型 ----------------
def build_tcn_multi_step(input_shape, output_steps=10):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, dilation_rate=1, padding='causal', activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'))
    model.add(Conv1D(64, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(output_steps))
    model.compile(optimizer='adam', loss='mae')
    return model

# ---------------- 時序 split ----------------
def time_series_split(X, y, test_ratio=0.15):
    n = len(X)
    test_n = int(n * test_ratio)
    split_idx = n - test_n
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# ---------------- conversions ----------------
def returns_to_future_close(last_close, pred_returns):
    closes = []
    c = float(last_close)
    for r in pred_returns:
        c = c * (1 + float(r))
        closes.append(c)
    return closes

# ---------------- metrics ----------------
def compute_metrics(y_true, y_pred):
    maes = []
    rmses = []
    for step in range(y_true.shape[1]):
        maes.append(mean_absolute_error(y_true[:, step], y_pred[:, step]))
        rmses.append(math.sqrt(mean_squared_error(y_true[:, step], y_pred[:, step])))
    return np.array(maes), np.array(rmses)

# ---------------- MA from predictions (vectorized convenience) ----------------
def compute_ma_from_predictions(last_known_window_closes, y_pred_matrix, ma_period=5):
    n_samples, window = last_known_window_closes.shape
    steps = y_pred_matrix.shape[1]
    preds_ma = np.zeros((n_samples, steps))
    for i in range(n_samples):
        seq = list(last_known_window_closes[i])  # copy
        for t in range(steps):
            seq.append(y_pred_matrix[i, t])
            look = seq[-ma_period:] if len(seq) >= ma_period else seq
            preds_ma[i, t] = np.mean(look)
    return preds_ma

def compute_true_ma(last_window, y_true, ma_period=5):
    n_samples, window = last_window.shape
    steps = y_true.shape[1]
    true_ma = np.zeros((n_samples, steps))
    for i in range(n_samples):
        seq = list(last_window[i])
        for t in range(steps):
            seq.append(y_true[i, t])
            look = seq[-ma_period:] if len(seq) >= ma_period else seq
            true_ma[i, t] = np.mean(look)
    return true_ma

# ---------------- plotting + upload (your version, slightly adjusted to accept df index) ----------------
def plot_and_upload_to_storage(df_real, df_future, bucket_obj=None, hist_days=10):
    df_real_plot = df_real.copy().tail(hist_days)
    if df_real_plot.empty:
        print("⚠️ df_real_plot 為空，無法繪圖")
        return None
    df_future = df_future.copy().reset_index(drop=True)
    last_hist_date = df_real_plot.index[-1]
    start_row = {
        "date": last_hist_date,
        "Pred_Close": df_real_plot['Close'].iloc[-1],
        "Pred_MA5": df_real_plot['SMA_5'].iloc[-1] if 'SMA_5' in df_real_plot.columns else df_real_plot['Close'].iloc[-1],
        "Pred_MA10": df_real_plot['SMA_10'].iloc[-1] if 'SMA_10' in df_real_plot.columns else df_real_plot['Close'].iloc[-1]
    }
    df_future_plot = pd.concat([pd.DataFrame([start_row]), df_future], ignore_index=True)
    plt.figure(figsize=(14,7))
    x_real = list(range(len(df_real_plot)))
    plt.plot(x_real, df_real_plot['Close'].values, label="Close")
    if 'SMA_5' in df_real_plot.columns:
        plt.plot(x_real, df_real_plot['SMA_5'].values, label="SMA5")
    if 'SMA_10' in df_real_plot.columns:
        plt.plot(x_real, df_real_plot['SMA_10'].values, label="SMA10")
    offset = len(df_real_plot) - 1
    x_future = [offset + i for i in range(len(df_future_plot))]
    plt.plot(x_future, df_future_plot['Pred_Close'].values, linestyle=':', marker='o', label="Pred Close")
    for xf, val in zip(x_future, df_future_plot['Pred_Close'].values):
        plt.annotate(
            f"{val:.2f}",
            xy=(xf, val),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=8,
            ha='left',
            va='bottom',
            color='red',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7)
        )
    plt.plot(x_future, df_future_plot['Pred_MA5'].values, linestyle='--', label="Pred MA5")
    plt.plot(x_future, df_future_plot['Pred_MA10'].values, linestyle='--', label="Pred MA10")
    labels = []
    for d in df_real_plot.index[:-1]:
        labels.append(pd.Timestamp(d).strftime('%m-%d'))
    for d in df_future_plot['date']:
        labels.append(pd.Timestamp(d).strftime('%m-%d'))
    ticks = list(range(len(labels)))
    plt.xticks(ticks=ticks, labels=labels, rotation=45)
    plt.xlim(0, max(ticks))
    plt.legend()
    plt.title("2301.TW 歷史 + 預測（近 {} 日 + 未來 {} 日）".format(hist_days, len(df_future_plot)-1))
    plt.xlabel("Date")
    plt.ylabel("Price")
    os.makedirs("results", exist_ok=True)
    file_name = f"{datetime.now().strftime('%Y-%m-%d')}_future_trade_days.png"
    file_path = os.path.join("results", file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("📌 圖片已儲存：", file_path)
    if bucket_obj is not None:
        try:
            blob = bucket_obj.blob(f"LSTM_Pred_Images/{file_name}")
            blob.upload_from_filename(file_path)
            try:
                blob.make_public()
                public_url = blob.public_url
            except Exception:
                public_url = blob.public_url if getattr(blob, 'public_url', None) else None
            print("🔥 圖片已上傳至 Storage：", public_url)
            return public_url
        except Exception as e:
            print("❌ 上傳 Storage 發生錯誤：", e)
            return None
    return None

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 60
    PRED_STEPS = 10
    PERIOD = "36mo"
    TEST_RATIO = 0.15

    # 1) 取得資料並計指標
    df = fetch_and_prepare(ticker=TICKER, period=PERIOD)
    df = update_today_from_firestore(df)  # optional update
    save_stock_data_to_firestore(df, ticker=TICKER)  # optional write historical

    # choose a compact set of features (含 price 作為一欄)
    feature_cols = [
        'Close','Volume','ret_lag_1','ret_lag_2','ret_lag_3',
        'EMA_5','EMA_10','EMA_20','EMA_50','MACD_hist',
        'RSI_14','StochRSI','ATR_14','BB_width','OBV','OBV_SMA_20',
        'vol_5','vol_10','day_of_week','is_month_end'
    ]
    # ensure all exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X, y, used_features = build_multi_step_returns(df[feature_cols + ['Close']], close_col='Close', pred_steps=PRED_STEPS, lookback=LOOKBACK, feature_cols=feature_cols)
    print("X shape:", X.shape, "y shape:", y.shape)

    # 2) time series split
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_ratio=TEST_RATIO)

    # 3) scaling
    nsamples, tw, nfeatures = X_train.shape
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train.reshape((-1, nfeatures)))
    def scale_X(X_raw):
        s = X_raw.reshape((-1, X_raw.shape[-1]))
        return scaler_x.transform(s).reshape(X_raw.shape)
    X_train_s = scale_X(X_train)
    X_test_s = scale_X(X_test)

    # scale y (returns) per column using MinMaxScaler on flattened 2D (n_samples x steps)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)  # shape (n_samples, pred_steps)
    y_train_s = scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    # 4) build & train TCN
    model = build_tcn_multi_step((LOOKBACK, nfeatures), output_steps=PRED_STEPS)
    model.summary()
    os.makedirs("models", exist_ok=True)
    ckpt_path = f"models/{TICKER}_tcn_best.h5"
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
    history = model.fit(X_train_s, y_train_s, validation_data=(X_test_s, y_test_s),
                        epochs=60, batch_size=32, callbacks=[es, mc], verbose=2)

    # 5) predict and inverse scale
    pred_s = model.predict(X_test_s)
    pred = scaler_y.inverse_transform(pred_s)  # returns in decimal (e.g., 0.01 == 1%)

    # compute metrics on returns (MAE / RMSE per step)
    maes_model, rmses_model = compute_metrics(y_test, pred)
    print("Per-step MAE (model returns):", np.round(maes_model, 6))
    print("Avg MAE (model returns):", np.round(maes_model.mean(), 6))

    # Baselines (last-close repeated -> convert to returns baseline relative to last_close)
    last_known_closes_all = X_test[:, -1, feature_cols.index('Close')] if 'Close' in feature_cols else X_test[:, -1, 0]
    # Baseline A: assume returns 0 (flat) -> predicted returns all zeros
    baselineA = np.zeros_like(pred)
    # Baseline B: repeat last 1-day return
    if 'ret_lag_1' in feature_cols:
        last_ret_1 = X_test[:, -1, feature_cols.index('ret_lag_1')]
        baselineB = np.zeros_like(pred)
        for i in range(baselineB.shape[0]):
            r = last_ret_1[i]
            price = last_known_closes_all[i]
            # build returns sequence by repeating r (geometric compounding)
            for t in range(baselineB.shape[1]):
                baselineB[i, t] = r
    else:
        baselineB = baselineA.copy()

    maes_bA, rmses_bA = compute_metrics(y_test, baselineA)
    maes_bB, rmses_bB = compute_metrics(y_test, baselineB)

    print("Avg MAE model returns:", np.round(maes_model.mean(),6),
          "baselineA:", np.round(maes_bA.mean(),6), "baselineB:", np.round(maes_bB.mean(),6))

    # 6) evaluate derived price MA errors: convert predictions -> closes and compare MA
    # build last_closes_window for each test sample (we need real last LOOKBACK closes)
    # Note: X_test contains scaled features; we need last raw closes from unscaled X_test slice: take from original X using index
    # We'll reconstruct last_closes from X_test (unscaled) using original X slices.
    last_closes_window = X_test[:, -LOOKBACK:, feature_cols.index('Close')] if 'Close' in feature_cols else X_test[:, -LOOKBACK:, 0]
    # For correct conversion we need the last observed actual close for each sample (the final element of last_closes_window)
    # But last_closes_window currently is unscaled (since X_test is original before scaling) — correct.
    # Convert model returns to future closes for each sample
    n_test = pred.shape[0]
    model_closes = np.zeros_like(pred)
    baselineA_closes = np.zeros_like(pred)
    baselineB_closes = np.zeros_like(pred)
    true_closes = np.zeros_like(pred)
    for i in range(n_test):
        last_close = last_closes_window[i, -1]
        # true future closes (from y_test: returns) -> convert cumulative
        true_returns = y_test[i]
        # compute true closes
        tc = []
        c = float(last_close)
        for r in true_returns:
            c = c * (1 + float(r))
            tc.append(c)
        true_closes[i, :] = tc
        # model predicted returns
        mr = pred[i]
        mc = returns_to_future_close(last_close, mr)
        model_closes[i, :] = mc
        # baselineA: returns = 0 -> flat price equal to last_close
        baselineA_closes[i, :] = [last_close for _ in range(pred.shape[1])]
        # baselineB: use baselineB returns if available -> convert
        br = baselineB[i]
        bc = returns_to_future_close(last_close, br)
        baselineB_closes[i, :] = bc

    # compute MAE of MA5 / MA10 of derived series
    model_MA5 = compute_ma_from_predictions(last_closes_window, model_closes, ma_period=5)
    model_MA10 = compute_ma_from_predictions(last_closes_window, model_closes, ma_period=10)
    bA_MA5 = compute_ma_from_predictions(last_closes_window, baselineA_closes, ma_period=5)
    bA_MA10 = compute_ma_from_predictions(last_closes_window, baselineA_closes, ma_period=10)
    true_MA5 = compute_true_ma(last_closes_window, true_closes, ma_period=5)
    true_MA10 = compute_true_ma(last_closes_window, true_closes, ma_period=10)

    mae_model_MA5 = np.mean(np.abs(model_MA5 - true_MA5))
    mae_bA_MA5 = np.mean(np.abs(bA_MA5 - true_MA5))
    mae_model_MA10 = np.mean(np.abs(model_MA10 - true_MA10))
    mae_bA_MA10 = np.mean(np.abs(bA_MA10 - true_MA10))

    print("MAE on derived MA5 -> model:", np.round(mae_model_MA5,4), "baselineA:", np.round(mae_bA_MA5,4))
    print("MAE on derived MA10 -> model:", np.round(mae_model_MA10,4), "baselineA:", np.round(mae_bA_MA10,4))

    # 7) generate final df_future for plotting using last window of full df
    # Use model trained on all (optionally you can retrain on full dataset)
    # We'll use the model and last LOOKBACK rows in df
    X_last_raw = df[feature_cols].tail(LOOKBACK).values.reshape(1, LOOKBACK, len(feature_cols))
    X_last_s = scale_X(X_last_raw)
    pred_last_s = model.predict(X_last_s)[0]
    pred_last_returns = scaler_y.inverse_transform(pred_last_s.reshape(1, -1))[0] if pred_last_s.ndim==1 else scaler_y.inverse_transform(pred_last_s)[0]
    last_close_real = df['Close'].iloc[-1]
    pred_last_closes = returns_to_future_close(last_close_real, pred_last_returns)

    # compute MA5/MA10 starting from last known closes (use last 10 real closes)
    last_known_closes = df['Close'].values[-10:].tolist()
    results = []
    seq = list(last_known_closes)
    for pc in pred_last_closes:
        seq.append(pc)
        ma5 = np.mean(seq[-5:]) if len(seq) >= 5 else np.mean(seq)
        ma10 = np.mean(seq[-10:]) if len(seq) >= 10 else np.mean(seq)
        results.append((pc, ma5, ma10))

    first_bday = (pd.Timestamp(df.index[-1].date()) + BDay(1)).date()
    business_days = pd.bdate_range(start=first_bday, periods=PRED_STEPS)
    df_future = pd.DataFrame({
        "date": business_days,
        "Pred_Close": [r[0] for r in results],
        "Pred_MA5": [r[1] for r in results],
        "Pred_MA10": [r[2] for r in results]
    })

    # 8) plot & upload
    image_url = plot_and_upload_to_storage(df, df_future, bucket_obj=bucket)
    print("Image URL:", image_url)
    print(df_future)

    # 9) write predictions to Firestore (if enabled)
    if db is not None:
        for i, row in df_future.iterrows():
            try:
                db.collection("NEW_stock_data_liteon_preds").document(row['date'].strftime("%Y-%m-%d")).set({
                    TICKER: {
                        "Pred_Close": float(row['Pred_Close']),
                        "Pred_MA5": float(row['Pred_MA5']),
                        "Pred_MA10": float(row['Pred_MA10'])
                    }
                })
            except Exception as e:
                print("寫入預測到 Firestore 發生錯誤：", e)
        try:
            pred_table_serialized = []
            for _, r in df_future.reset_index(drop=True).iterrows():
                rec = {
                    "date": pd.Timestamp(r['date']).strftime("%Y-%m-%d"),
                    "Pred_Close": float(r['Pred_Close']),
                    "Pred_MA5": float(r['Pred_MA5']),
                    "Pred_MA10": float(r['Pred_MA10'])
                }
                pred_table_serialized.append(rec)
            meta_doc = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "image_url": image_url,
                "pred_table": pred_table_serialized,
                "update_time": datetime.now().isoformat()
            }
            db.collection("NEW_stock_data_liteon_preds_meta").document(datetime.now().strftime("%Y-%m-%d")).set(meta_doc)
        except Exception as e:
            print("寫入預測 metadata 到 Firestore 發生錯誤：", e)
        print("🔥 預測寫入 Firestore 完成")
