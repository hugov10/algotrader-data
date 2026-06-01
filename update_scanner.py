#!/usr/bin/env python3
"""
AlgoTrader AR v14 — Scanner Automático
Descarga datos de yfinance, calcula indicadores y genera scanner_data.json
Ejecutado por GitHub Actions cada 1 hora (horario BYMA: 11:00-17:00 AR)
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════
PANEL_TICKERS = [
    "ALUA.BA", "BYMA.BA", "MIRG.BA", "EDN.BA", "SUPV.BA", "CADO.BA", "INVJ.BA",
    "BBAR.BA", "LOMA.BA", "TECO2.BA", "PAMP.BA", "TGNO4.BA", "TGSU2.BA",
    "CEPU.BA", "COME.BA", "TRAN.BA", "METR.BA", "AGRO.BA", "IRSA.BA",
    "LEDE.BA", "MOLI.BA", "MORI.BA", "SAMI.BA", "CELU.BA", "FERR.BA",
    "HARG.BA", "AUSO.BA", "CTIO.BA", "BHIP.BA", "VALO.BA", "CRES.BA",
    "YPFD.BA", "BMA.BA", "GGAL.BA", "TXAR.BA", "BBAR.BA"
]

CEDEAR_TICKERS = [
    "AAPL.BA", "MSFT.BA", "GOOGL.BA", "META.BA", "NVDA.BA", "TSLA.BA",
    "AMZN.BA", "NFLX.BA", "AMD.BA", "INTC.BA", "QCOM.BA", "CRM.BA",
    "PYPL.BA", "MELI.BA", "GLOB.BA", "TSM.BA", "KO.BA", "PEP.BA",
    "MCD.BA", "WMT.BA", "JPM.BA", "V.BA", "XOM.BA", "CVX.BA",
    "JNJ.BA", "PFE.BA", "VALE.BA", "GOLD.BA", "SPY.BA", "QQQ.BA",
    "GLD.BA", "SLV.BA", "ARKK.BA", "SONY.BA", "LLY.BA", "AMGN.BA",
    "UBER.BA", "BRKB.BA", "GS.BA"
]

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calc_bollinger(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def analyze_ticker(ticker, is_cedear=False):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="60d")
        if len(hist) < 30:
            return None

        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']

        price = float(close.iloc[-1])
        price_prev = float(close.iloc[-2])
        chg_pct = ((price - price_prev) / price_prev) * 100

        # Indicadores
        rsi = float(calc_rsi(close).iloc[-1])
        rsi_prev = float(calc_rsi(close).iloc[-2])

        macd_line, signal_line = calc_macd(close)
        macd_val = float(macd_line.iloc[-1])
        signal_val = float(signal_line.iloc[-1])
        macd = "bullish" if macd_val > signal_val else "bearish" if macd_val < signal_val else "neutral"

        upper, middle, lower = calc_bollinger(close)
        if price > float(upper.iloc[-1]):
            bb = "upper"
        elif price < float(lower.iloc[-1]):
            bb = "lower"
        else:
            bb = "middle"

        # Tendencia
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        trend = "uptrend" if ema20.iloc[-1] > ema50.iloc[-1] else "downtrend" if ema20.iloc[-1] < ema50.iloc[-1] else "neutral"

        # Volumen
        vol = int(volume.iloc[-1])
        vol_sma20 = volume.rolling(20).mean().iloc[-1]
        vol_ratio = float(vol / vol_sma20) if vol_sma20 > 0 else 1.0

        # ATR y niveles
        atr = float(calc_atr(high, low, close).iloc[-1])
        sl = price - (atr * 1.5)
        tp1 = price + (atr * 2.5)
        tp2 = price + (atr * 4.5)
        sl_pct = abs((sl - price) / price) * 100

        # Soporte/Resistencia (mínimos/máximos recientes)
        sup = float(low.rolling(20).min().iloc[-1])
        res = float(high.rolling(20).max().iloc[-1])

        # Score algorítmico (0-100)
        score = 50
        if 30 <= rsi <= 70:
            score += 10
        elif rsi < 30:
            score += 20
        elif rsi > 70:
            score -= 15

        if macd == "bullish":
            score += 15
        elif macd == "bearish":
            score -= 10

        if trend == "uptrend":
            score += 15
        elif trend == "downtrend":
            score -= 10

        if vol_ratio >= 2.0:
            score += 15
        elif vol_ratio >= 1.5:
            score += 10
        elif vol_ratio >= 1.0:
            score += 5
        else:
            score -= 5

        if bb == "lower" and trend != "downtrend":
            score += 10
        elif bb == "upper" and vol_ratio >= 1.5:
            score += 5

        if chg_pct > 2:
            score += 5
        elif chg_pct < -2:
            score -= 5

        score = max(0, min(100, int(score)))

        # Recomendación
        if score >= 75:
            rec = "STRONG_BUY"
        elif score >= 60:
            rec = "BUY"
        elif score >= 40:
            rec = "HOLD"
        elif score >= 25:
            rec = "SELL"
        else:
            rec = "STRONG_SELL"

        return {
            "sym": ticker.replace(".BA", ""),
            "price": round(price, 2),
            "chgPct": round(chg_pct, 2),
            "vol": vol,
            "volRatio": round(vol_ratio, 2),
            "rsi": round(rsi, 1),
            "rsiPrev": round(rsi_prev, 1),
            "macd": macd,
            "trend": trend,
            "bb": bb,
            "score": score,
            "rec": rec,
            "sl": round(sl, 2),
            "sl_pct": round(sl_pct, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "sup": round(sup, 2),
            "res": round(res, 2),
            "atr": round(atr, 2),
            "bid": round(price * 0.998, 2),
            "ask": round(price * 1.002, 2)
        }
    except Exception as e:
        print(f"Error en {ticker}: {e}")
        return None

def main():
    now = datetime.now()

    print(f"🚀 Iniciando scanner — {now.strftime('%Y-%m-%d %H:%M')}")

    # Analizar Panel
    panel_data = []
    for t in PANEL_TICKERS:
        result = analyze_ticker(t)
        if result:
            panel_data.append(result)
            print(f"  ✅ {t}: Score {result['score']} | {result['rec']}")
        else:
            print(f"  ❌ {t}: Sin datos")

    # Analizar CEDEARS
    cedear_data = []
    for t in CEDEAR_TICKERS:
        result = analyze_ticker(t, is_cedear=True)
        if result:
            cedear_data.append(result)
            print(f"  ✅ {t}: Score {result['score']} | {result['rec']}")
        else:
            print(f"  ❌ {t}: Sin datos")

    # Sentimiento
    panel_bull = sum(1 for s in panel_data if s['rec'] in ['BUY', 'STRONG_BUY'])
    panel_bear = sum(1 for s in panel_data if s['rec'] in ['SELL', 'STRONG_SELL'])
    panel_sent = "bullish" if panel_bull > panel_bear else "bearish" if panel_bear > panel_bull else "neutral"

    cedear_bull = sum(1 for s in cedear_data if s['rec'] in ['BUY', 'STRONG_BUY'])
    cedear_bear = sum(1 for s in cedear_data if s['rec'] in ['SELL', 'STRONG_SELL'])
    cedear_sent = "bullish" if cedear_bull > cedear_bear else "bearish" if cedear_bear > cedear_bull else "neutral"

    # Ordenar por score
    panel_data.sort(key=lambda x: x['score'], reverse=True)
    cedear_data.sort(key=lambda x: x['score'], reverse=True)

    output = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M"),
        "fecha": now.strftime("%d/%m/%Y"),
        "hora": now.strftime("%H:%M"),
        "panel": {
            "data": panel_data,
            "count": len(panel_data),
            "sentiment": panel_sent
        },
        "cedear": {
            "data": cedear_data,
            "count": len(cedear_data),
            "sentiment": cedear_sent
        }
    }

    with open("scanner_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Scanner completo:")
    print(f"   Panel: {len(panel_data)} activos | Sentimiento: {panel_sent}")
    print(f"   CEDEARS: {len(cedear_data)} activos | Sentimiento: {cedear_sent}")
    print(f"   Guardado en scanner_data.json")

if __name__ == "__main__":
    main()
