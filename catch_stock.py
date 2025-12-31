# -*- coding: utf-8 -*-
"""
個股資料抓取 + 技術指標計算 + Firestore 更新與寫回
✅ B 方案：第一次初始化寫完整歷史，其後只寫最近 N 天
✅ 今日 Close 先覆寫，再重新計算指標
✅ 指數 / 外生因子只寫最近交易日
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

# ================== 參數 ==================
WRITE_DAYS = 3
COLLECTION = "NEW_stock_data_liteon"
PERIOD = "12mo"
INIT_CHECK_TICKER = "2301.TW"   # 用來判斷是否初始化

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
    print("⚠️ FIREBASE 未設定，Firestore 寫入將略過")

# ================= 初始化判斷 =================
def is_init_mode(ticker: str) -> bool:
    """
    Firestore 中找不到任何含 ticker 的 document → 視為第一次初始化
    """
    if db is None:
        return False

    docs = (
        db.collection(COLLECTION)
        .limit(1)
        .stream()
    )

    for doc in docs:
        if ticker in doc.to_dict():
            return False

    return True

# ================= 交易日工具 =================
def get_last_trading_day(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return None, False

    last_day = df.index[-1].normalize()
    today = pd.Timestamp(datetime.now().date())
    return last_day, last_day == today

# ================= 技術指標 =================
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rs = gain.rolling(20).mean() / loss.rolling(20).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    denom = high14 - low14
    df["K"] = np.where(denom == 0, 50, 100 * (df["Close"] - low14) / denom)
    df["D"] = df["K"].rolling(3).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["SignalLine"] = df["MACD"].ewm(span=9, adjust=False).mean()

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    return df.dropna()

# ================= 覆寫最近交易日 Close =================
def overwrite_last_close(df, ticker):
    if db is None or df is None or len(df) == 0:
        return df

    last_day, is_today_trading = get_last_trading_day(df)
    date_str = last_day.strftime("%Y-%m-%d")

    doc = db.collection(COLLECTION).document(date_str).get()
    if doc.exists:
        payload = doc.to_dict().get(ticker, {})
        if "Close" in payload:
            df.loc[last_day, "Close"] = float(payload["Close"])

    return df

# ================= 個股流程 =================
def fetch_prepare_recalc(ticker):
    df = yf.Ticker(ticker).history(period=PERIOD)
    df = overwrite_last_close(df, ticker)
    return add_all_indicators(df)

def save_stock(df, ticker, init_mode=False):
    if db is None:
        return

    df_write = df if init_mode else df.tail(WRITE_DAYS)
    batch = db.batch()

    for idx, row in df_write.iterrows():
        doc_ref = db.collection(COLLECTION).document(idx.strftime("%Y-%m-%d"))
        batch.set(doc_ref, {
            ticker: {
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
        }, merge=True)

    batch.commit()

    if init_mode:
        print(f"🚀 {ticker} 初始化完成（{len(df_write)} 天）")
    else:
        print(f"🔥 {ticker} 更新最近 {len(df_write)} 天")

# ================= 指數 / 外生因子 =================
def save_factor_latest(tickers, alias):
    if db is None:
        return

    for tk in tickers:
        try:
            df = yf.Ticker(tk).history(period=PERIOD)
            if len(df) == 0:
                continue

            last_day, _ = get_last_trading_day(df)
            row = df.loc[last_day]

            db.collection(COLLECTION).document(
                last_day.strftime("%Y-%m-%d")
            ).set({
                alias: {"Close": float(row["Close"])}
            }, merge=True)

            return
        except Exception:
            continue

# ================= Main =================
if __name__ == "__main__":

    INIT_MODE = is_init_mode(INIT_CHECK_TICKER)

    if INIT_MODE:
        print("🚀 偵測為第一次初始化，將寫入完整歷史資料")
    else:
        print("🔁 一般更新模式（只寫最近資料）")

    for ticker in ["2301.TW", "2408.TW", "8110.TW"]:
        df = fetch_prepare_recalc(ticker)
        save_stock(df, ticker, init_mode=INIT_MODE)

    save_factor_latest(["^TWII"], "TAIEX")
    save_factor_latest(["^TELI", "IR0027.TW"], "ELECTRONICS")
    save_factor_latest(["^SOX", "SOXX", "SMH"], "SOX")
    save_factor_latest(["MU", "MU.VI", "MU.MX"], "MU_US")
    save_factor_latest(["TWD=X", "USDTWD=X"], "USD_TWD")

    print("✅ 全部完成")
