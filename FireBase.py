# -*- coding: utf-8 -*-
"""
改良版：光寶科（2301.TW）多步 LSTM -> 預測未來 10 個交易日 Close，再計算 MA5/MA10
重點改進：
 - 預測目標改為未來 10 日 Close（multi-step）
 - 新增技術特徵（returns, ATR, Bollinger, OBV, SMA diffs）
 - 時序 train/test split（避免資料洩漏）
 - EarlyStopping / ModelCheckpoint / 更好的 scaler 使用
 - 評估使用 MAE / RMSE，並以預測 closes 計算 Pred MA5/MA10 做最終比對
 - 圖表只顯示交易日，預測線與歷史線連接
"""
import os, json
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import math

# ---------------- Firebase 初始化 ----------------
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    db = None
    print("⚠️ FIREBASE env 未設定 — 會略過上傳步驟")

# ---------------- 特徵工程函式 ----------------
def add_technical_features(df):
    df = df.copy()
    # SMA
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()

    # returns & log returns
    df['RET_1'] = df['Close'].pct_change().fillna(0)
    df['LOG_RET_1'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)

    # SMA diffs
    df['Close_minus_SMA5'] = df['Close'] - df['SMA_5']
    df['SMA5_minus_SMA10'] = df['SMA_5'] - df['SMA_10']

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']

    # OBV (On Balance Volume)
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

    # Volume moving average
    df['Vol_SMA_5'] = df['Volume'].rolling(5).mean()
    df['Vol_SMA_20'] = df['Volume'].rolling(20).mean()

    # fill / drop
    df = df.dropna()
    return df

# ---------------- 取得資料並計指標 ----------------
def fetch_and_prepare(ticker="2301.TW", period="12mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = add_technical_features(df)
    return df

# ---------------- 更新今天 Close 從 Firestore（若有） ----------------
def update_today_from_firestore(df):
    if db is None:
        return df
    today_str = datetime.now().strftime("%Y-%m-%d")
    doc_ref = db.collection("NEW_stock_data_liteon").document(today_str)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict().get("2301.TW", {})
        if "Close" in data:
            try:
                df.loc[pd.Timestamp(today_str), 'Close'] = float(data["Close"])
            except Exception:
                pass
    df = df.dropna()
    return df

# ---------------- 儲存到 Firestore（選用） ----------------
def save_to_firestore(df):
    if db is None:
        print("跳過 Firestore 寫入（未設定）")
        return
    selected = ['Close', 'MACD', 'RSI', 'K', 'D', 'Volume'] if 'MACD' in df.columns else ['Close', 'Volume']
    collection = "NEW_stock_data_liteon"
    batch = db.batch()
    count = 0
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        data = {col: float(row[col]) for col in selected if col in row and not pd.isna(row[col])}
        doc_ref = db.collection(collection).document(date_str)
        batch.set(doc_ref, {"2301.TW": data})
        count += 1
        if count >= 300:
            batch.commit()
            batch = db.batch()
            count = 0
    batch.commit()
    print("🔥 Firestore 寫入完成")

# ---------------- 建資料集（用 sliding window） ----------------
def create_sequences(df, features, target_steps=10, window=60):
    X, y = [], []
    closes = df['Close'].values
    data = df[features].values
    for i in range(window, len(df) - target_steps + 1):
        X.append(data[i-window:i])
        y.append(closes[i:i+target_steps])
    return np.array(X), np.array(y)

# ---------------- 建模型（multi-step LSTM） ----------------
def build_lstm_multi_step(input_shape, output_steps=10):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(output_steps))
    model.compile(optimizer='adam', loss='mae')
    return model

# ---------------- 時序 train/test split ----------------
def time_series_split(X, y, test_ratio=0.15):
    n = len(X)
    test_n = int(n * test_ratio)
    split_idx = n - test_n
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# ---------------- 從預測 closes 計算 MA5 / MA10 ----------------
def compute_pred_ma_from_pred_closes(last_known_closes, pred_closes):
    closes_seq = list(last_known_closes)
    results = []
    for pc in pred_closes:
        closes_seq.append(pc)
        ma5 = np.mean(closes_seq[-5:]) if len(closes_seq) >= 5 else np.mean(closes_seq)
        ma10 = np.mean(closes_seq[-10:]) if len(closes_seq) >= 10 else np.mean(closes_seq)
        results.append((pc, ma5, ma10))
    return results

# ---------------- 畫圖函式（交易日版，預測線連接歷史線） ----------------
def plot_all(df_real, df_future, hist_days=60):
    df_real = df_real.copy()
    df_real['date'] = pd.to_datetime(df_real.index).tz_localize(None)
    df_plot_real = df_real.tail(hist_days)
    df_future = df_future.copy()
    df_future['date'] = pd.to_datetime(df_future['date'])

    # 將最後一天歷史加入預測線開頭
    last_hist_date = df_plot_real['date'].iloc[-1]
    last_hist_close = df_plot_real['Close'].iloc[-1]
    last_sma5 = df_plot_real['SMA_5'].iloc[-1] if 'SMA_5' in df_plot_real.columns else last_hist_close
    last_sma10 = df_plot_real['SMA_10'].iloc[-1] if 'SMA_10' in df_plot_real.columns else last_hist_close
    df_future_plot = pd.concat([
        pd.DataFrame([{
            "date": last_hist_date,
            "Pred_Close": last_hist_close,
            "Pred_MA5": last_sma5,
            "Pred_MA10": last_sma10
        }]),
        df_future
    ], ignore_index=True)

    plt.figure(figsize=(16,8))
    plt.plot(df_plot_real['date'], df_plot_real['Close'], label="Close")
    if 'SMA_5' in df_plot_real.columns:
        plt.plot(df_plot_real['date'], df_plot_real['SMA_5'], label="SMA5")
    if 'SMA_10' in df_plot_real.columns:
        plt.plot(df_plot_real['date'], df_plot_real['SMA_10'], label="SMA10")

    plt.plot(df_future_plot['date'], df_future_plot['Pred_Close'], ':', label='Pred Close')
    plt.plot(df_future_plot['date'], df_future_plot['Pred_MA5'], '--', label="Pred MA5")
    plt.plot(df_future_plot['date'], df_future_plot['Pred_MA10'], '--', label="Pred MA10")

    plt.xticks(df_future_plot['date'], [d.strftime('%m-%d') for d in df_future_plot['date']], rotation=45)
    plt.legend()
    plt.title("2301.TW 歷史 + 預測（交易日）")
    plt.xlabel("Date")
    plt.ylabel("Price")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    file_path = f"{results_dir}/{today_str}_future_trade_days.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("📌 圖片已儲存：", file_path)

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 60
    PRED_STEPS = 10
    PERIOD = "18mo"
    TEST_RATIO = 0.15

    df = fetch_and_prepare(ticker=TICKER, period=PERIOD)
    df = update_today_from_firestore(df)

    features = ['Close', 'Volume', 'RET_1', 'LOG_RET_1', 'Close_minus_SMA5',
                'SMA5_minus_SMA10', 'ATR_14', 'BB_width', 'OBV', 'OBV_SMA_20',
                'Vol_SMA_5']
    df_features = df[features].dropna()

    X, y = create_sequences(df_features, features, target_steps=PRED_STEPS, window=LOOKBACK)
    print("X shape:", X.shape, "y shape:", y.shape)
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_ratio=TEST_RATIO)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Scaler
    nsamples, tw, nfeatures = X_train.shape
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train.reshape((nsamples*tw, nfeatures)))
    def scale_X(X_raw):
        s = X_raw.reshape((-1, X_raw.shape[-1]))
        return scaler_x.transform(s).reshape(X_raw.shape)
    X_train_s, X_test_s = scale_X(X_train), scale_X(X_test)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)
    y_train_s, y_test_s = scaler_y.transform(y_train), scaler_y.transform(y_test)

    model = build_lstm_multi_step(input_shape=(LOOKBACK, nfeatures), output_steps=PRED_STEPS)
    model.summary()

    os.makedirs("models", exist_ok=True)
    ckpt_path = f"models/{TICKER}_best.h5"
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(X_train_s, y_train_s, validation_data=(X_test_s, y_test_s),
                        epochs=80, batch_size=32, callbacks=[es, mc], verbose=2)

    pred_s = model.predict(X_test_s)
    pred = scaler_y.inverse_transform(pred_s)

    # 評估
    maes, rmses = [], []
    for step in range(PRED_STEPS):
        y_true, y_pred = y_test[:, step], pred[:, step]
        maes.append(mean_absolute_error(y_true, y_pred))
        rmses.append(math.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE per step:", np.round(maes,4))
    print("RMSE per step:", np.round(rmses,4))
    print("Avg MAE:", np.round(np.mean(maes),4))

    last_known_window = X_test[-1]
    last_known_closes = list(last_known_window[:,0])
    results = compute_pred_ma_from_pred_closes(last_known_closes, pred[-1])

    today = pd.Timestamp(datetime.now().date())
    first_bday = (today + BDay(1)).date()
    business_days = pd.bdate_range(start=first_bday, periods=PRED_STEPS)
    df_future = pd.DataFrame({
        "date": business_days,
        "Pred_Close": [r[0] for r in results],
        "Pred_MA5": [r[1] for r in results],
        "Pred_MA10": [r[2] for r in results]
    })

    plot_all(df, df_future, hist_days=60)
    print(df_future)

    # 寫入 Firestore
    if db is not None:
        for i, row in df_future.iterrows():
            db.collection("NEW_stock_data_liteon_preds").document(row['date'].strftime("%Y-%m-%d")).set({
                "2301.TW": {
                    "Pred_Close": float(row['Pred_Close']),
                    "Pred_MA5": float(row['Pred_MA5']),
                    "Pred_MA10": float(row['Pred_MA10'])
                }
            })
        print("🔥 預測寫入 Firestore 完成")
