# -*- coding: utf-8 -*-
"""
整合版：光寶科 LSTM 股價預測 + 預測 5/10 日線
🔥 功能：
  - 抓取股價
  - 計算技術指標
  - 寫入 Firestore
  - 訓練 LSTM
  - 預測未來 MA5 與 MA10
  - 畫圖顯示
"""

import os, json
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ============================ 🔐 Firebase 初始化 ============================
key_dict = json.loads(os.environ["FIREBASE"])
cred = credentials.Certificate(key_dict)

try:
    firebase_admin.get_app()
except:
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ============================ 📌 抓股票 + 計算指標 ============================
def fetch_and_calculate():
    ticker_symbol = "2301.TW"
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="6mo")

    # 技術指標計算
    df['SMA_5'] = df['Close'].rolling(window=5).mean().round(5)
    df['SMA_10'] = df['Close'].rolling(window=10).mean().round(5)

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    df['RSI'] = (100 - (100 / (1 + (gain.rolling(20).mean() / loss.rolling(20).mean())))).round(5)

    df['Lowest_14'] = df['Low'].rolling(window=14).min()
    df['Highest_14'] = df['High'].rolling(window=14).max()
    df['K'] = (100 * (df['Close'] - df['Lowest_14']) / (df['Highest_14'] - df['Lowest_14'])).round(5)
    df['D'] = df['K'].rolling(window=3).mean().round(5)

    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (df['EMA_12'] - df['EMA_26']).round(5)

    # 計算未來 MA5、MA10 作為 LSTM 預測目標
    df['Fut_MA5'] = df['Close'].rolling(5).mean().shift(-4)
    df['Fut_MA10'] = df['Close'].rolling(10).mean().shift(-9)

    return df.dropna()


# ============================ 💾 寫入 Firestore ============================
def save_to_firestore(df):
    selected = ['Close', 'MACD', 'RSI', 'K', 'D', 'Volume']
    collection = "NEW_stock_data_liteon"

    batch = db.batch()
    count = 0

    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        data = {col: float(row[col]) for col in selected if not pd.isna(row[col])}
        doc_ref = db.collection(collection).document(date_str)
        batch.set(doc_ref, {"2301.TW": data})
        count += 1

        if count >= 300:
            batch.commit()
            batch = db.batch()

    batch.commit()
    print("🔥 Firestore 寫入完成")


# ============================ 🤖 建 LSTM 模型 ============================
def train_lstm(df):
    features = ['Close', 'Volume', 'MACD', 'RSI', 'K', 'D']
    targets = ['Fut_MA5', 'Fut_MA10']

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[targets])

    window = 30
    X, y = [], []

    for i in range(window, len(df)):
        X.append(X_scaled[i-window:i])
        y.append(y_scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(2)   # 預測 MA5 和 MA10
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=32, verbose=1)

    print("🎉 LSTM 訓練完成")
    return model, scaler_x, scaler_y, X_scaled


# ============================ 🔮 預測未來 MA5 / MA10 ============================
def predict_future_ma(model, scaler_x, scaler_y, X_scaled, df):
    last_30 = X_scaled[-30:]
    future = []

    for _ in range(10):
        pred = model.predict(last_30.reshape(1, 30, X_scaled.shape[1]))
        future.append(pred[0])

        # 更新 last_30
        new_row = np.zeros((1, X_scaled.shape[1]))
        new_row[0, 0] = pred[0][0]  # Close 位置可用 MA5 代替
        last_30 = np.append(last_30[1:], new_row, axis=0)

    future_array = np.array(future)
    future_ma = scaler_y.inverse_transform(future_array)

    # 使用 fetch_and_calculate() 回傳的 df，直接從最後日期推算
    last_date = pd.to_datetime(df.index[-1])
    dates = [last_date + timedelta(days=i) for i in range(1, 11)]

    df_future = pd.DataFrame({
        "date": dates,
        "Pred_MA5": future_ma[:, 0],
        "Pred_MA10": future_ma[:, 1]
    })

    return df_future


# ============================ 📈 畫圖 ============================
# ============================ 📈 畫圖 ============================
def plot_all(df_real, df_future):
    # 將 index 當作日期
    df_real['date'] = pd.to_datetime(df_real.index)
    df_future['date'] = pd.to_datetime(df_future['date'])

    # 合併歷史與預測日期範圍
    df_all = pd.concat([df_real[['date','Close','SMA_5','SMA_10']], df_future.set_index('date')], axis=0)

    plt.figure(figsize=(12,6))

    # 畫歷史收盤價
    plt.plot(df_real['date'], df_real['Close'], label="Close", color="blue")

    # 畫歷史 SMA5 / SMA10
    plt.plot(df_real['date'], df_real['SMA_5'], label="SMA5", color="green")
    plt.plot(df_real['date'], df_real['SMA_10'], label="SMA10", color="orange")

    # 畫預測 MA5 / MA10
    plt.plot(df_future['date'], df_future['Pred_MA5'], '--', label="Pred MA5", color="lime")
    plt.plot(df_future['date'], df_future['Pred_MA10'], '--', label="Pred MA10", color="red")

    # 設定 X 軸以日為單位
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.gcf().autofmt_xdate(rotation=45)  # 斜顯示日期

    plt.legend()
    plt.title("2301.TW 預測 5/10 日線（每日刻度）")
    plt.xlabel("Date")
    plt.ylabel("Price")

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    today = datetime.now().strftime("%Y-%m-%d")
    file_path = f"{results_dir}/{today}.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📌 圖片已儲存：{file_path}")


# ============================ ▶️ 主流程 ============================
if __name__ == "__main__":
    df = fetch_and_calculate()          # 抓股價 + 計算指標
    save_to_firestore(df)               # 寫入 Firestore

    # 直接用 df 訓練 LSTM，避免 KeyError
    model, scaler_x, scaler_y, X_scaled = train_lstm(df)

    df_future = predict_future_ma(model, scaler_x, scaler_y, X_scaled, df)
    plot_all(df, df_future)
