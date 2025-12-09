# -*- coding: utf-8 -*-
"""
整合版：光寶科 LSTM 股價預測 + 5/10 日線繪製
🔥 功能：
  - 抓取股價
  - 計算技術指標
  - 寫入 Firestore
  - 訓練 LSTM
  - 預測未來 10 天
  - 計算 SMA_5 與 SMA_10
  - 畫圖顯示
"""

import os, json
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
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

    return df


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


# ============================ 📥 Firestore 讀取 ============================
def read_from_firestore():
    docs = db.collection("NEW_stock_data_liteon").stream()

    rows = []
    for doc in docs:
        data = doc.to_dict().get("2301.TW", {})
        data["date"] = doc.id
        rows.append(data)

    df = pd.DataFrame(rows).sort_values("date")
    df.reset_index(drop=True, inplace=True)
    return df


# ============================ 🤖 建 LSTM 模型 ============================
def train_lstm(df):
    features = ['Close', 'Volume', 'MACD', 'RSI', 'K', 'D']

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = [], []
    window = 30

    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i][0])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=32, verbose=1)

    print("🎉 LSTM 訓練完成")
    return model, scaler, scaled


# ============================ 🔮 預測未來 10 天 ============================
def predict_future(model, scaler, scaled, df):
    last_30 = scaled[-30:]     # shape (30, 6)
    future = []

    for _ in range(10):
        pred = model.predict(last_30.reshape(1, 30, scaled.shape[1]))
        future.append(pred[0][0])

        # 🔥 修正：把 pred 擴展成 shape (1, 6)
        pred_full = np.zeros((1, scaled.shape[1]))
        pred_full[0, 0] = pred[0][0]  # Close 位置

        # 🔥 正確拼接
        last_30 = np.append(last_30[1:], pred_full, axis=0)

    future_array = np.array(future).reshape(-1, 1)
    zeros_array = np.zeros((future_array.shape[0], scaled.shape[1] - 1))
    stacked = np.hstack((future_array, zeros_array))

    future_prices = scaler.inverse_transform(stacked)[:, 0]

    # 避免 pandas "closed" 參數警告（新版已移除）
    dates = pd.date_range(df['date'].iloc[-1], periods=11)[1:]

    df_future = pd.DataFrame({
        "date": dates,
        "Close": future_prices
    })

    return df_future


# ============================ 📈 畫圖 ============================
def plot_all(df_real, df_future):
    df_all = pd.concat([df_real[['date','Close']], df_future])

    # 🔥 這一行很重要：統一日期格式
    df_all['date'] = pd.to_datetime(df_all['date'])

    df_all['SMA_5'] = df_all['Close'].rolling(5).mean()
    df_all['SMA_10'] = df_all['Close'].rolling(10).mean()

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    today = datetime.now().strftime("%Y-%m-%d")
    file_path = f"{results_dir}/{today}.png"

    plt.figure(figsize=(12,6))
    plt.plot(df_all['date'], df_all['Close'], label="Real/Pred Close")
    plt.plot(df_all['date'], df_all['SMA_5'], label="SMA 5")
    plt.plot(df_all['date'], df_all['SMA_10'], label="SMA 10")
    plt.legend()
    plt.title("2301.TW 預測 + 5/10 日線")

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📌 圖片已儲存：{file_path}")


# ============================ ▶️ 主流程 ============================
if __name__ == "__main__":
    df = fetch_and_calculate()          # 抓股價 + 指標
    save_to_firestore(df)               # 寫入 Firestore

    df_train = read_from_firestore()    # 讀 Firestore
    model, scaler, scaled = train_lstm(df_train)

    df_future = predict_future(model, scaler, scaled, df_train)
    plot_all(df_train, df_future)
