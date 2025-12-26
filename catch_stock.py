# -*- coding: utf-8 -*-
"""
個股資料抓取 + 技術指標計算 + Firestore 更新與寫回
✅ 今日 Close 先覆寫，再重新計算指標（一致性修正版）
✅ 改1：個股只寫最近 N 天（預設 3 天）
✅ 改2：指數 / 外生因子只寫最新一天
✅ 改3：自動判斷是否為交易日，非交易日改用最近交易日
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

# ================= 交易日工具 =================
def get_last_trading_day(df: pd.DataFrame):
    """
    回傳 (last_trading_day: Timestamp, is_today_trading: bool)
    """
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

    if not is_today_trading:
        print(f"ℹ️ 今日非交易日，{ticker} 改用最近交易日 {date_str}")

    doc = db.collection(COLLECTION).document(date_str).get()
    if doc.exists:
        payload = doc.to_dict().get(ticker, {})
        if "Close" in payload:
            df.loc[last_day, "Close"] = float(payload["Close"])
            print(f"✔ 覆寫 {ticker} Close ({date_str}): {payload['Close']}")

    return df

# ================= 個股流程 =================
def fetch_prepare_recalc(ticker):
    df = yf.Ticker(ticker).history(period=PERIOD)
    df = overwrite_last_close(df, ticker)
    return add_all_indicators(df)

def save_stock_recent_days(df, ticker):
    if db is None:
        return

    df_tail = df.tail(WRITE_DAYS)
    batch = db.batch()

    for idx, row in df_tail.iterrows():
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
    print(f"🔥 {ticker} 寫入最近 {len(df_tail)} 天")

# ================= 指數 / 外生因子（只寫最近交易日） =================
def save_factor_latest(tickers, alias):
    if db is None:
        return

    for tk in tickers:
        try:
            df = yf.Ticker(tk).history(period=PERIOD)
            if len(df) == 0:
                continue

            last_day, is_today_trading = get_last_trading_day(df)
            row = df.loc[last_day]
            date_str = last_day.strftime("%Y-%m-%d")

            if not is_today_trading:
                print(f"ℹ️ 今日非交易日，{alias} 使用 {date_str}")

            db.collection(COLLECTION).document(date_str).set({
                alias: {"Close": float(row["Close"])}
            }, merge=True)

            print(f"🔥 {alias} 更新成功（來源 {tk}）")
            return

        except Exception:
            continue

    print(f"⚠️ {alias} 全部來源失敗")

# ================= Main =================
if __name__ == "__main__":

    for ticker in ["2301.TW", "2408.TW", "8110.TW"]:
        df = fetch_prepare_recalc(ticker)
        save_stock_recent_days(df, ticker)

    save_factor_latest(["^TWII"], "TAIEX")
    save_factor_latest(["^TELI", "IR0027.TW"], "ELECTRONICS")
    save_factor_latest(["^SOX", "SOXX", "SMH"], "SOX")
    save_factor_latest(["MU", "MU.VI", "MU.MX"], "MU_US")
    save_factor_latest(["TWD=X", "USDTWD=X"], "USD_TWD")

    print("✅ 全部完成")
