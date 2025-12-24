# -*- coding: utf-8 -*-
"""
個股資料抓取 + 技術指標計算 + Firestore 更新與寫回
✅ 今日 Close 先覆寫，再重新計算指標（一致性修正版）
➕ 加入加權指數 / 電子指數（Close only）
➕ 加入南亞科 2408.TW（同方法：覆寫今日 Close → 重算指標 → 寫回）
➕ 加入華東 8110.TW（同方法：覆寫今日 Close → 重算指標 → 寫回）
✅ NEW：加入外生因子（Close only）
   - SOX（費半）^SOX（抓不到則 fallback SOXX/SMH）
   - MU（美光）MU（抓不到則 fallback MU.VI / MU.MX）
   - USD/TWD（匯率）優先嘗試 TWD=X（找不到就換備援代碼）
不含模型、不含預測、不含繪圖
"""

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------- Firebase 初始化 ----------------
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
    print("⚠️ FIREBASE 未設定，Firestore 寫入將略過")

# ---------------- 技術指標計算（全集中） ----------------
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # SMA
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    # RSI(20)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(20).mean()
    avg_loss = loss.rolling(20).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # KD(14,3)
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    denom = high14 - low14
    df["K"] = np.where(denom == 0, 50.0, 100 * (df["Close"] - low14) / denom)
    df["D"] = df["K"].rolling(3).mean()

    # MACD(12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["SignalLine"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Returns
    df["RET_1"] = df["Close"].pct_change()
    df["LOG_RET_1"] = np.log(df["Close"] / df["Close"].shift(1))

    # ATR(14)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # Bollinger Bands(20,2)
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_mid"] = mid
    df["BB_upper"] = mid + 2 * std
    df["BB_lower"] = mid - 2 * std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / mid

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_SMA_20"] = df["OBV"].rolling(20).mean()

    # Volume SMA
    df["Vol_SMA_5"] = df["Volume"].rolling(5).mean()
    df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()

    return df.dropna()

# ---------------- Firestore 覆寫今日 Close ----------------
def overwrite_today_close(df: pd.DataFrame, ticker: str, collection: str = "NEW_stock_data_liteon") -> pd.DataFrame:
    if db is None:
        return df

    today = datetime.now().strftime("%Y-%m-%d")
    try:
        doc = db.collection(collection).document(today).get()
        if doc.exists:
            payload = doc.to_dict().get(ticker, {})
            if isinstance(payload, dict) and "Close" in payload:
                ts = pd.Timestamp(today)
                if ts in df.index:
                    df.loc[ts, "Close"] = float(payload["Close"])
                    print(f"✔ Firestore 覆寫今日 Close ({ticker})：{payload['Close']}")
    except Exception as e:
        print(f"⚠️ 今日 Close 覆寫失敗 ({ticker})：{e}")

    return df

# ---------------- 抓個股 ----------------
def fetch_prepare_recalc(ticker: str = "2301.TW", period: str = "12mo", collection: str = "NEW_stock_data_liteon") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    df = overwrite_today_close(df, ticker, collection=collection)
    df = add_all_indicators(df)
    return df

# ---------------- Firestore 寫個股 ----------------
def save_to_firestore(df: pd.DataFrame, ticker: str = "2301.TW", collection: str = "NEW_stock_data_liteon"):
    if db is None:
        return

    batch = db.batch()
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        payload = {
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": float(row["Volume"]),
            "MACD": float(row["MACD"]),
            "RSI": float(row["RSI"]),
            "K": float(row["K"]),
            "D": float(row["D"]),
            "ATR_14": float(row["ATR_14"]),
        }
        doc_ref = db.collection(collection).document(date_str)
        batch.set(doc_ref, {ticker: payload}, merge=True)

    batch.commit()
    print(f"🔥 Firestore 寫入完成：{ticker}")

# ---------------- ➕ 指數/外生因子抓取（Close only） ----------------
def _fetch_history_with_fallback(tickers, period="12mo"):
    last_err = None
    for tk in tickers:
        try:
            df = yf.Ticker(tk).history(period=period)
            if df is not None and len(df) > 0 and "Close" in df.columns:
                return tk, df
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"⚠️ 無法抓取資料：{tickers} | last_err={last_err}")

def save_index_close(ticker: str, alias: str, period: str = "12mo", collection: str = "NEW_stock_data_liteon"):
    if db is None:
        return

    df = yf.Ticker(ticker).history(period=period)
    if df is None or len(df) == 0:
        print(f"⚠️ 指數/因子無資料：{ticker}（略過 {alias}）")
        return

    for idx, row in df.iterrows():
        date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")
        doc_ref = db.collection(collection).document(date_str)
        doc_ref.set({alias: {"Close": float(row["Close"])}}, merge=True)

    print(f"🔥 指數/因子寫入完成：{alias}")

def save_factor_close_with_fallback(tickers, alias: str, period: str = "12mo", collection: str = "NEW_stock_data_liteon"):
    if db is None:
        return

    used, df = _fetch_history_with_fallback(tickers, period=period)

    for idx, row in df.iterrows():
        date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")
        doc_ref = db.collection(collection).document(date_str)
        doc_ref.set({alias: {"Close": float(row["Close"])}}, merge=True)

    print(f"🔥 指數/因子寫入完成：{alias}（來源：{used}）")

# ---------------- Main ----------------
if __name__ == "__main__":
    COLLECTION = "NEW_stock_data_liteon"
    PERIOD = "12mo"

    # 2301：光寶科
    df_2301 = fetch_prepare_recalc("2301.TW", period=PERIOD, collection=COLLECTION)
    save_to_firestore(df_2301, "2301.TW", collection=COLLECTION)

    # 2408：南亞科（同方法）
    df_2408 = fetch_prepare_recalc("2408.TW", period=PERIOD, collection=COLLECTION)
    save_to_firestore(df_2408, "2408.TW", collection=COLLECTION)

    # 8110：華東（同方法）
    df_8110 = fetch_prepare_recalc("8110.TW", period=PERIOD, collection=COLLECTION)
    save_to_firestore(df_8110, "8110.TW", collection=COLLECTION)

    # ➕ 加權指數（Close only）
    save_index_close("^TWII", "TAIEX", period=PERIOD, collection=COLLECTION)

    # ✅ 電子類指數（先試 ^TELI，不行再 IR0027.TW）
    save_factor_close_with_fallback(
        tickers=["^TELI", "IR0027.TW"],
        alias="ELECTRONICS",
        period=PERIOD,
        collection=COLLECTION,
    )

    # ✅ 外生因子（Close only）— 改成 fallback，避免抓不到就整段沒資料
    # 1) 半導體 proxy：^SOX → SOXX → SMH
    save_factor_close_with_fallback(
        tickers=["^SOX", "SOXX", "SMH"],
        alias="SOX",
        period=PERIOD,
        collection=COLLECTION,
    )

    # 2) 美光：MU → MU.VI → MU.MX
    save_factor_close_with_fallback(
        tickers=["MU", "MU.VI", "MU.MX"],
        alias="MU_US",
        period=PERIOD,
        collection=COLLECTION,
    )

    # 3) USD/TWD（匯率）
    save_factor_close_with_fallback(
        tickers=["TWD=X", "USDTWD=X", "USD/TWD", "USDTWD"],
        alias="USD_TWD",
        period=PERIOD,
        collection=COLLECTION,
    )

    print("2301 tail:\n", df_2301.tail())
    print("2408 tail:\n", df_2408.tail())
    print("8110 tail:\n", df_8110.tail())
