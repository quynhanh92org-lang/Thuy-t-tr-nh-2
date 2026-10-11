import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
import os
import re
import math
import warnings
import io
import glob
import unicodedata
from itertools import product
from datetime import datetime

warnings.filterwarnings("ignore")

# ==============================================================================
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="HOSE V4 — Nested Walk-Forward Quant Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện Fintech hiện đại & chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .tech-badge {
        background-color: #F1F5F9;
        color: #1E293B;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #CBD5E1;
    }
    .tech-badge-primary {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }
    .tech-badge-success {
        background-color: #ECFDF5;
        color: #047857;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #A7F3D0;
    }
    .kpi-card {
        background: #FFFFFF;
        padding: 16px 18px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
        transition: all 0.2s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .kpi-title {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
        margin: 4px 0;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94A3B8;
    }
    .status-box-champ {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #3B82F6;
        border-left: 6px solid #1D4ED8;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .status-box-warn {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 1px solid #F59E0B;
        border-left: 6px solid #D97706;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HẰNG SỐ & THIẾT LẬP MẶC ĐỊNH THEO CHUẨN HOSE V4
# ==============================================================================
TRADING_DAYS = 252
BENCHMARK_ALIASES = {"VNINDEX", "VN-INDEX", "VNI", "^VNINDEX", "VNINDEXINDEX"}
DEFAULT_TOP_N = 5
DEFAULT_INITIAL_CASH = 1_000_000_000.0
DEFAULT_COMMISSION = 0.0015
DEFAULT_SLIPPAGE = 0.0005
DEFAULT_SELECTION_FREQ = "Q"       # Tái chọn Top N theo quý
DEFAULT_ALLOCATION_FREQ = "M"      # Cập nhật trọng số theo tháng
DEFAULT_WEIGHT_LOOKBACK = 252
DEFAULT_CORR_LOOKBACK = 126
DEFAULT_MAX_PAIRWISE_CORR = 0.80
DEFAULT_MIN_TA_SCORE_AT_SELECTION = 4

ACTIVE_MODEL_SPECS = {
    "DYN_EW_REGIME":      {"family": "EW",        "regime": True, "mask": True, "name": "Dynamic EW + Regime"},
    "DYN_MPT_LW_REGIME":  {"family": "MPT_LW",    "regime": True, "mask": True, "name": "Dynamic MPT-LW + Regime"},
    "DYN_MINVAR_REGIME":  {"family": "MINVAR_LW", "regime": True, "mask": True, "name": "Dynamic MinVar-LW + Regime"},
    "DYN_HRP_REGIME":     {"family": "HRP",       "regime": True, "mask": True, "name": "Dynamic HRP + Regime"},
    "DYN_BLEND50_REGIME": {"family": "BLEND50",   "regime": True, "mask": True, "name": "Dynamic Blend50 + Regime"},
}

# ==============================================================================
# 1. HÀM ĐỌC FILE VỚI CƠ CHẾ AUTO-FALLBACK ENCODING (KHẮC PHỤC TRIỆT ĐỂ LỖI UTF-8 0xE3)
# ==============================================================================
def load_data_file(source):
    """
    Đọc file CSV hoặc Excel với cơ chế tự động thử nhiều bảng mã:
    UTF-8, UTF-8-sig, Latin-1, CP1252, ISO-8859-1, CP1258.
    Ngăn chặn 100% lỗi UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3...
    """
    # 1. Kiểm tra nếu là file Excel
    is_excel = False
    if hasattr(source, "name") and (str(source.name).endswith(".xlsx") or str(source.name).endswith(".xls")):
        is_excel = True
    elif isinstance(source, str) and (source.endswith(".xlsx") or source.endswith(".xls")):
        is_excel = True

    if is_excel:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_excel(source)
        except Exception:
            pass

    # 2. Đọc file CSV với danh sách Encoding đa tầng
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1", "cp1258"]
    for enc in encodings:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding=enc)
            if df is not None and len(df) > 0 and len(df.columns) > 1:
                return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue

    # 3. Fallback cuối cùng: UTF-8 với chế độ thay thế ký tự lỗi
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source, encoding="utf-8", encoding_errors="replace")

def clean_col_name(s):
    s = str(s).strip()
    decomposed = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z0-9]", "", stripped.lower())

def find_col_smart(df, candidates):
    cleaned_cols = {clean_col_name(c): c for c in df.columns}
    raw_cols = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}

    for cand in candidates:
        cand_clean = clean_col_name(cand)
        if cand_clean in cleaned_cols:
            return cleaned_cols[cand_clean]
        if cand_clean in raw_cols:
            return raw_cols[cand_clean]
        for clean_c, original_c in cleaned_cols.items():
            if cand_clean in clean_c:
                return original_c
    return None

def parse_dates_safely(s):
    s = pd.Series(s).astype(str).str.strip()
    x = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    if x.notna().mean() < 0.80:
        x = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if x.notna().mean() < 0.80:
        x = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return x

def clean_numeric_series(s):
    if s is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s_clean = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s_clean, errors="coerce")

def standardize_ohlcv(df_raw, use_adjusted=True):
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Tìm các cột bằng hàm thông minh hỗ trợ cả Tiếng Việt và Tiếng Anh
    ticker_col = find_col_smart(df, ["ticker", "symbol", "code", "stock", "mack", "ma", "m"])
    date_col = find_col_smart(df, ["date", "datetime", "tradingdate", "time", "ngay", "ngy"])
    volume_col = find_col_smart(df, ["volume", "vol", "tradingvolume", "khoiluongkhoplenh", "khoiluong", "khoplenh", "kl"])

    raw_open = find_col_smart(df, ["open", "giamocua", "mocua", "mo", "gimc", "giam"])
    raw_high = find_col_smart(df, ["high", "giacaonhat", "caonhat", "cao", "giacao"])
    raw_low = find_col_smart(df, ["low", "giathapnhat", "thapnhat", "thap", "giathap"])
    raw_close = find_col_smart(df, ["close", "giadongcua", "dongcua", "dong", "giaongca"])

    adj_open = find_col_smart(df, ["adjopen", "adjustedopen", "adj_open"])
    adj_high = find_col_smart(df, ["adjhigh", "adjustedhigh", "adj_high"])
    adj_low = find_col_smart(df, ["adjlow", "adjustedlow", "adj_low"])
    adj_close = find_col_smart(df, ["adjclose", "adjustedclose", "adj_close", "giadieuchinh", "dieuchinh", "giaieuchnh"])

    # Cơ chế vị trí dự phòng (đặc biệt phù hợp chuẩn dữ liệu 12 cột sàn HOSE/CafeF/Vietstock)
    cols = list(df.columns)
    if len(cols) >= 11:
        if ticker_col is None:
            ticker_col = cols[0]
        if date_col is None:
            date_col = cols[1]
        if raw_close is None:
            raw_close = cols[2]
        if adj_close is None:
            adj_close = cols[3]
        if volume_col is None:
            volume_col = cols[5]
        if raw_open is None:
            raw_open = cols[9]
        if raw_high is None:
            raw_high = cols[10]
        if raw_low is None:
            raw_low = cols[11]

    if ticker_col is None or date_col is None:
        if len(cols) >= 2:
            ticker_col = cols[0]
            date_col = cols[1]
        else:
            raise ValueError(f"Không thể xác định cột Ticker và Date trong file. Các cột hiện có: {list(df.columns)}")

    has_all_adj = all(v is not None and v in df.columns for v in [adj_open, adj_high, adj_low, adj_close])
    has_adj_close = adj_close is not None and adj_close in df.columns

    open_s = clean_numeric_series(df[adj_open]) if (has_all_adj and use_adjusted) else (clean_numeric_series(df[raw_open]) if raw_open else None)
    high_s = clean_numeric_series(df[adj_high]) if (has_all_adj and use_adjusted) else (clean_numeric_series(df[raw_high]) if raw_high else None)
    low_s = clean_numeric_series(df[adj_low]) if (has_all_adj and use_adjusted) else (clean_numeric_series(df[raw_low]) if raw_low else None)
    close_raw_s = clean_numeric_series(df[raw_close]) if (raw_close and raw_close in df.columns) else None
    close_adj_s = clean_numeric_series(df[adj_close]) if (has_adj_close and adj_close in df.columns) else None

    close_s = close_adj_s if (use_adjusted and close_adj_s is not None) else (close_raw_s if close_raw_s is not None else close_adj_s)

    if close_s is None:
        raise ValueError("Không tìm thấy dữ liệu giá Đóng cửa hợp lệ.")

    if open_s is None:
        open_s = close_s.copy()
    if high_s is None:
        high_s = close_s.copy()
    if low_s is None:
        low_s = close_s.copy()

    # Nếu dùng giá điều chỉnh nhưng file chỉ có Giá điều chỉnh ở cột Close, nhân tỷ lệ điều chỉnh cho OHLC
    if use_adjusted and not has_all_adj and close_adj_s is not None and close_raw_s is not None:
        ratio = (close_adj_s / close_raw_s.replace(0, np.nan)).fillna(1.0).clip(0.01, 100.0)
        open_s = open_s * ratio
        high_s = high_s * ratio
        low_s = low_s * ratio

    vol_s = clean_numeric_series(df[volume_col]) if (volume_col and volume_col in df.columns) else pd.Series(0.0, index=df.index)

    out = pd.DataFrame({
        "Ticker": df[ticker_col].astype(str).str.strip().str.upper(),
        "Date": parse_dates_safely(df[date_col]),
        "Open": open_s,
        "High": high_s,
        "Low": low_s,
        "Close": close_s,
        "Volume": vol_s.fillna(0.0).clip(lower=0.0)
    })

    out = out.dropna(subset=["Ticker", "Date", "Open", "High", "Low", "Close"])
    out = out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]

    out["High"] = out[["High", "Open", "Close"]].max(axis=1)
    out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)

    out = out.drop_duplicates(["Ticker", "Date"], keep="last")
    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return out

def data_quality_table(data):
    rows = []
    for t, g in data.groupby("Ticker"):
        rows.append({
            "Ticker": t,
            "Start": g["Date"].min().strftime("%Y-%m-%d"),
            "End": g["Date"].max().strftime("%Y-%m-%d"),
            "Sessions": int(g["Date"].nunique()),
            "MissingVolumePct": float(100.0 * g["Volume"].isna().mean()),
            "MedianTradedValue": float((g["Close"] * g["Volume"]).median()),
        })
    return pd.DataFrame(rows).sort_values("Sessions", ascending=False).reset_index(drop=True)

# ==============================================================================
# 2. CHIA TẬP DỮ LIỆU CHỐNG FUTURE LEAKAGE
# ==============================================================================
def resolve_auto_splits(data, split_mode="auto", train_prop=0.6, val_prop=0.2, test_prop=0.2,
                        min_total_sessions=420, min_core_train=252, min_val=63, min_test=100,
                        prefer_last_year=True, last_year_min_sessions=180):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(data["Date"].unique())))
    n = len(dates)
    if n < min_total_sessions:
        raise ValueError(f"Dữ liệu có {n} phiên (cần tối thiểu {min_total_sessions} phiên để chia Core Train, Validation và Test)")

    if split_mode == "proportions":
        i_train = max(min_core_train, int(n * train_prop))
        i_val_end = max(i_train + min_val, int(n * (train_prop + val_prop)))
        i_val_end = min(i_val_end, n - min_test)
        core_end = dates[i_train - 1]
        val_start, val_end = dates[i_train], dates[i_val_end - 1]
        test_start, test_end = dates[i_val_end], dates[-1]
        return dates, core_end, val_start, val_end, test_start, test_end

    last_year = dates[-1].year
    last_year_dates = dates[dates.year == last_year]
    pre_last_year_count = int((dates < last_year_dates[0]).sum()) if len(last_year_dates) else 0

    if prefer_last_year and len(last_year_dates) >= last_year_min_sessions and pre_last_year_count >= min_core_train + min_val:
        test_start, test_end = last_year_dates[0], last_year_dates[-1]
        pretest = dates[dates < test_start]
    else:
        test_n = max(min_test, min(252, int(round(test_prop * n))))
        test_start, test_end = dates[-test_n], dates[-1]
        pretest = dates[dates < test_start]

    val_n = max(min_val, min(252, int(round(0.25 * len(pretest)))))
    val_n = min(val_n, len(pretest) - min_core_train)
    val_start, val_end = pretest[-val_n], pretest[-1]
    core_end = pretest[-val_n - 1]
    return dates, core_end, val_start, val_end, test_start, test_end

# ==============================================================================
# 3. TÍNH CHỈ BÁO KỸ THUẬT & TA SCORE (0–10)
# ==============================================================================
def ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi_wilder(close, window=14):
    d = close.diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    ag = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    al = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = 100 - 100/(1 + rs)
    rsi.loc[(al == 0) & (ag > 0)] = 100
    rsi.loc[(ag == 0) & (al > 0)] = 0
    return rsi

def atr_wilder(df, window=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

def adx_wilder(df, window=14):
    high, low = df["High"], df["Low"]
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = atr_wilder(df, window)
    plus_di = 100 * plus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    return adx, plus_di, minus_di

def calculate_ta_features(df):
    out = df.copy().sort_index()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)

    ema12, ema26 = ema(out["Close"], 12), ema(out["Close"], 26)
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = ema(out["MACD"], 9)

    out["ADX"], out["Plus_DI"], out["Minus_DI"] = adx_wilder(out, 14)
    out["ATR"] = atr_wilder(out, 14)
    out["ATR_pct"] = 100 * out["ATR"] / out["Close"].replace(0, np.nan)
    out["ATR_pct_Q90"] = out["ATR_pct"].rolling(126, min_periods=40).quantile(0.90)

    out["Volume_MA20"] = out["Volume"].rolling(20, min_periods=20).mean()
    out["Volume_Ratio"] = out["Volume"] / out["Volume_MA20"].replace(0, np.nan)
    out["OBV"] = (np.sign(out["Close"].diff()).fillna(0) * out["Volume"]).cumsum()
    out["OBV_EMA20"] = ema(out["OBV"], 20)

    out["RSI"] = rsi_wilder(out["Close"], 14)
    out["ROC10"] = out["Close"].pct_change(10) * 100
    out["High_52W"] = out["High"].rolling(252, min_periods=150).max()
    out["Close_to_52W_High"] = out["Close"] / out["High_52W"].replace(0, np.nan)
    out["Avg_Traded_Value_60"] = (out["Close"] * out["Volume"]).rolling(60, min_periods=20).mean()

    out["Momentum63"] = out["Close"].pct_change(63)
    out["Momentum126"] = out["Close"].pct_change(126)
    out["Momentum252"] = out["Close"].pct_change(252)
    out["Vol63"] = out["Close"].pct_change().rolling(63, min_periods=40).std() * np.sqrt(TRADING_DAYS)
    rollmax = out["Close"].rolling(126, min_periods=60).max()
    out["DD126"] = out["Close"] / rollmax - 1

    # TA Score 0–10 chuẩn V4 Group 1
    out["Trend_Score"] = (
        (out["Close"] > out["EMA50"]).astype(int) * 2 +
        (out["EMA50"] > out["EMA200"]).astype(int) * 2 +
        (out["MACD"] > out["MACD_Signal"]).astype(int)
    ).clip(0, 5)

    momentum_core = ((out["RSI"] >= 45) & (out["RSI"] <= 75)) | (out["ROC10"] > 0)
    out["Momentum_Score"] = (
        momentum_core.astype(int) * 2 +
        (out["Close_to_52W_High"] >= 0.85).astype(int)
    ).clip(0, 3)

    out["Volume_Score"] = ((out["Volume_Ratio"] >= 1.0) | (out["OBV"] > out["OBV_EMA20"])).astype(int)
    out["Volatility_Score"] = (
        out["ATR_pct"].notna() & out["ATR_pct_Q90"].notna() &
        (out["ATR_pct"] <= out["ATR_pct_Q90"])
    ).astype(int)

    out["TA_Score"] = (
        out["Trend_Score"] + out["Momentum_Score"] +
        out["Volume_Score"] + out["Volatility_Score"]
    ).clip(0, 10).astype(int)
    return out

def stateful_active_from_score(score, buy_threshold, exit_threshold):
    state = 0
    vals = []
    for v in score.fillna(-np.inf):
        if v >= buy_threshold:
            state = 1
        elif v <= exit_threshold:
            state = 0
        vals.append(state)
    return pd.Series(vals, index=score.index, dtype=int)

# ==============================================================================
# 4. THỐNG KÊ HIỆU SUẤT & SIMULATION MỘT MÃ
# ==============================================================================
def annualized_sharpe(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    return np.sqrt(TRADING_DAYS) * r.mean() / r.std(ddof=1) if len(r) > 1 and r.std(ddof=1) > 1e-12 else np.nan

def annualized_sortino(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    neg = r[r < 0]
    return np.sqrt(TRADING_DAYS) * r.mean() / neg.std(ddof=1) if len(neg) > 1 and neg.std(ddof=1) > 1e-12 else np.nan

def metrics_from_equity(eq):
    eq = pd.Series(eq).dropna()
    if len(eq) < 2:
        return {}
    r = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 365.25)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(r) > 1 else np.nan
    return {
        "Return [%]": total * 100,
        "CAGR [%]": cagr * 100,
        "Sharpe": annualized_sharpe(r),
        "Sortino": annualized_sortino(r),
        "Volatility [%]": vol * 100 if pd.notna(vol) else np.nan,
        "MaxDD [%]": mdd * 100 if pd.notna(mdd) else np.nan,
        "Calmar": cagr / abs(mdd) if pd.notna(mdd) and mdd < 0 else np.nan,
    }

def simulate_single_stock(features, start, end, buy_threshold, exit_threshold, initial_cash=1_000_000.0, commission=DEFAULT_COMMISSION, slippage=DEFAULT_SLIPPAGE):
    active_close = stateful_active_from_score(features["TA_Score"], buy_threshold, exit_threshold)
    active_open = active_close.shift(1).fillna(0).astype(int)
    x = features.loc[(features.index >= pd.Timestamp(start)) & (features.index <= pd.Timestamp(end))].copy()
    active_open = active_open.reindex(x.index).fillna(0).astype(int)
    if x.empty:
        return {}

    cash, shares, state, trades = float(initial_cash), 0.0, 0, 0
    eq = []
    for dt, row in x.iterrows():
        d = int(active_open.loc[dt])
        op, cl = float(row["Open"]), float(row["Close"])
        if d != state:
            if d == 1:
                px = op * (1 + slippage)
                qty = math.floor(cash / (px * (1 + commission)))
                if qty > 0:
                    cash -= qty * px * (1 + commission)
                    shares = qty
                    state = 1
                    trades += 1
            else:
                px = op * (1 - slippage)
                cash += shares * px * (1 - commission)
                shares = 0
                state = 0
                trades += 1
        eq.append((dt, cash + shares * cl))

    m = metrics_from_equity(pd.Series(dict(eq)).sort_index())
    m["Trades"] = trades
    return m

# ==============================================================================
# 5. UNIVERSE BUILDER & TỐI ƯU HÓA NGƯỠNG TA TRÊN CORE TRAIN
# ==============================================================================
def build_universe(price_dict, asof, benchmark_ticker=None, min_history_floor=180, min_history_cap=400, liquidity_lookback=60, liquidity_min_percentile=0.20, use_liquidity=True):
    all_dates = set()
    for px in price_dict.values():
        all_dates.update(px.loc[:asof].index)
    pre_sessions = len(all_dates)
    min_hist = min(min_history_cap, max(min_history_floor, int(0.55 * pre_sessions)))

    rows = []
    for t, px in price_dict.items():
        if t == benchmark_ticker:
            continue
        h = px.loc[:asof]
        if h.empty:
            continue
        avg_value = (h["Close"] * h["Volume"]).tail(liquidity_lookback).mean()
        rows.append({
            "Ticker": t,
            "HistorySessions": len(h),
            "AvgTradedValue": avg_value,
            "EligibleHistory": len(h) >= min_hist,
        })

    u = pd.DataFrame(rows)
    if u.empty:
        return u
    ok = u["EligibleHistory"] & u["AvgTradedValue"].notna()
    u.loc[ok, "LiquidityPercentile"] = u.loc[ok, "AvgTradedValue"].rank(pct=True)
    u["PassLiquidity"] = (u["LiquidityPercentile"].fillna(0) >= liquidity_min_percentile) if use_liquidity else True
    u["Candidate"] = u["EligibleHistory"] & u["PassLiquidity"]
    return u.sort_values("AvgTradedValue", ascending=False).reset_index(drop=True)

def make_equal_folds(dates, start, end, n_folds=3):
    d = pd.DatetimeIndex(dates)
    d = d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    if len(d) < n_folds * 20:
        n_folds = max(1, len(d) // 20)
    chunks = np.array_split(d, n_folds)
    folds = []
    for i, c in enumerate(chunks, 1):
        if len(c) >= 20:
            folds.append({"Fold": f"F{i}", "start": pd.Timestamp(c[0]), "end": pd.Timestamp(c[-1])})
    return folds

def choose_global_ta_thresholds(feature_store, candidates, folds, buy_choices=[5, 6, 7], exit_choices=[2, 3, 4], penalty=0.20):
    trials = []
    for b, e in product(buy_choices, exit_choices):
        if e >= b:
            continue
        stock_scores = []
        for t in candidates:
            ms = []
            for f in folds:
                m = simulate_single_stock(feature_store[t], f["start"], f["end"], b, e)
                if m:
                    sh = m.get("Sharpe", np.nan)
                    sh = -3.0 if not np.isfinite(sh) else float(sh)
                    ms.append((sh, float(m.get("Return [%]", 0)), float(m.get("MaxDD [%]", -100))))
            if ms:
                a = np.array(ms, float)
                stock_scores.append([np.nanmean(a[:, 0]), np.nanmean(a[:, 1]), np.nanmin(a[:, 2])])
        if not stock_scores:
            continue

        z = np.array(stock_scores, float)
        med_sh = float(np.nanmedian(z[:, 0]))
        sd_sh = float(np.nanstd(z[:, 0]))
        med_ret = float(np.nanmedian(z[:, 1]))
        med_dd = float(np.nanmedian(z[:, 2]))
        score = med_sh - penalty * sd_sh + 0.005 * med_ret + 0.003 * med_dd
        trials.append({
            "BuyThreshold": b,
            "ExitThreshold": e,
            "GlobalCVScore": score,
            "MedianSharpe": med_sh,
            "SharpeStd": sd_sh,
            "MedianReturnPct": med_ret,
            "MedianMaxDDPct": med_dd,
            "NStocks": len(stock_scores),
        })

    tr = pd.DataFrame(trials).sort_values("GlobalCVScore", ascending=False).reset_index(drop=True)
    if tr.empty:
        return 5, 2, tr
    best = tr.iloc[0]
    return int(best["BuyThreshold"]), int(best["ExitThreshold"]), tr

# ==============================================================================
# 6. NESTED WALK-FORWARD XẾP HẠNG & TÁI CHỌN TOP N
# ==============================================================================
def trailing_stock_quality(features, asof, b, e, lookback=126):
    h = features.loc[:asof]
    if len(h) < 40:
        return {"TrailingSharpe": -3.0, "TrailingReturnPct": -100.0, "TrailingMaxDDPct": -100.0}
    sub_idx = h.index[-min(lookback, len(h)):]
    m = simulate_single_stock(features, sub_idx[0], sub_idx[-1], b, e)
    sh = m.get("Sharpe", np.nan) if m else np.nan
    return {
        "TrailingSharpe": -3.0 if not np.isfinite(sh) else float(sh),
        "TrailingReturnPct": float(m.get("Return [%]", -100.0)) if m else -100.0,
        "TrailingMaxDDPct": float(m.get("MaxDD [%]", -100.0)) if m else -100.0,
    }

def rank_and_select_asof(feature_store, price_dict, benchmark_features, benchmark_ticker,
                         asof, b, e, top_n=DEFAULT_TOP_N, min_ta_score=DEFAULT_MIN_TA_SCORE_AT_SELECTION,
                         max_vol_p=0.75, min_dd_p=0.15, max_rsi=80.0, max_corr=DEFAULT_MAX_PAIRWISE_CORR):
    universe = build_universe(price_dict, asof, benchmark_ticker)
    if universe.empty:
        raise ValueError(f"Universe rỗng tại {asof}")
    candidates = universe.loc[universe["Candidate"], "Ticker"].tolist()
    bench = benchmark_features.loc[:asof]
    bench_m126 = bench["Close"].pct_change(126).iloc[-1] if len(bench) > 126 else np.nan
    uidx = universe.set_index("Ticker")

    rows = []
    for t in candidates:
        feat = feature_store[t].loc[:asof]
        if feat.empty:
            continue
        snap = feat.iloc[-1]
        tq = trailing_stock_quality(feature_store[t], asof, b, e)
        rows.append({
            "Ticker": t,
            **tq,
            "TA_Score": float(snap.get("TA_Score", np.nan)),
            "Momentum63": float(snap.get("Momentum63", np.nan)),
            "Momentum126": float(snap.get("Momentum126", np.nan)),
            "Momentum252": float(snap.get("Momentum252", np.nan)),
            "RelativeStrength126": (float(snap.get("Momentum126", np.nan) - bench_m126) if pd.notna(bench_m126) else float(snap.get("Momentum126", np.nan))),
            "Vol63": float(snap.get("Vol63", np.nan)),
            "RSI": float(snap.get("RSI", np.nan)),
            "LiquidityPercentile": float(uidx.loc[t, "LiquidityPercentile"]),
        })

    r = pd.DataFrame(rows)
    if r.empty:
        raise ValueError(f"Không có mã nào đủ điều kiện tại {asof}")

    r["PassTA"] = r["TA_Score"].fillna(0) >= min_ta_score
    if int(r["PassTA"].sum()) >= top_n:
        r = r[r["PassTA"]].copy()

    vol_cap = r["Vol63"].quantile(max_vol_p)
    dd_floor = r["TrailingMaxDDPct"].quantile(min_dd_p)
    r["PassRiskGate"] = (
        (r["Vol63"].isna() | (r["Vol63"] <= vol_cap)) &
        (r["TrailingMaxDDPct"].isna() | (r["TrailingMaxDDPct"] >= dd_floor)) &
        (r["RSI"].isna() | (r["RSI"] <= max_rsi))
    )
    if int(r["PassRiskGate"].sum()) >= top_n:
        r = r[r["PassRiskGate"]].copy()

    r["P_Trailing"] = r["TrailingSharpe"].rank(pct=True)
    r["P_TA"] = r["TA_Score"].rank(pct=True)
    r["P_RS"] = r["RelativeStrength126"].rank(pct=True)
    r["P_LowVol"] = (-r["Vol63"]).rank(pct=True)
    r["P_Liq"] = r["LiquidityPercentile"].rank(pct=True)
    mommix = 0.6 * r["Momentum126"].fillna(r["Momentum63"]) + 0.4 * r["Momentum63"].fillna(0)
    r["P_Momentum"] = mommix.rank(pct=True)

    r["TA_Composite"] = (
        0.25 * r["P_Trailing"] +
        0.20 * r["P_TA"] +
        0.25 * r["P_RS"] +
        0.10 * r["P_LowVol"] +
        0.05 * r["P_Liq"] +
        0.15 * r["P_Momentum"]
    ) * 100

    r["FinalScore"] = r["TA_Composite"]
    r = r.sort_values(["FinalScore", "TrailingSharpe"], ascending=False).reset_index(drop=True)

    corr_tickers = r["Ticker"].tolist()
    close_list = [feature_store[t].loc[:asof, "Close"].rename(t) for t in corr_tickers]
    corr_px = pd.concat(close_list, axis=1, join="inner").tail(DEFAULT_CORR_LOOKBACK + 1) if close_list else pd.DataFrame()
    corr = corr_px.pct_change(fill_method=None).dropna().corr() if len(corr_px) > 20 else pd.DataFrame()

    selected = []
    for _, row in r.iterrows():
        t = row["Ticker"]
        if selected and not corr.empty:
            too_close = any(
                t in corr.index and x in corr.columns and
                pd.notna(corr.loc[t, x]) and abs(corr.loc[t, x]) > max_corr
                for x in selected
            )
            if too_close:
                continue
        selected.append(t)
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for t in r["Ticker"]:
            if t not in selected:
                selected.append(t)
            if len(selected) >= top_n:
                break

    locked = r[r["Ticker"].isin(selected)].copy()
    locked["SelectedOrder"] = locked["Ticker"].map({t: i + 1 for i, t in enumerate(selected)})
    locked["AsOfDate"] = pd.Timestamp(asof)
    locked = locked.sort_values("SelectedOrder")
    return r, locked, selected

def first_session_each_period(calendar, start, end, freq="Q"):
    idx = pd.DatetimeIndex(calendar)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(idx) == 0:
        return []
    s = pd.Series(idx, index=idx)
    key = idx.to_period(freq)
    firsts = s.groupby(key).first().tolist()
    if idx[0] not in firsts:
        firsts = [idx[0]] + firsts
    return sorted(set(pd.Timestamp(x) for x in firsts))

def previous_session(calendar, dt):
    idx = pd.DatetimeIndex(calendar)
    prev = idx[idx < pd.Timestamp(dt)]
    return pd.Timestamp(prev[-1]) if len(prev) else None

def build_selection_schedule(feature_store, price_dict, benchmark_ticker, benchmark_features,
                             calendar, start, end, b, e, selection_freq=DEFAULT_SELECTION_FREQ, top_n=DEFAULT_TOP_N):
    effective_dates = first_session_each_period(calendar, start, end, selection_freq)
    rows = []
    details = {}

    for eff in effective_dates:
        asof = previous_session(calendar, eff)
        if asof is None:
            continue
        ranking, locked, selected = rank_and_select_asof(
            feature_store, price_dict, benchmark_features, benchmark_ticker,
            asof, b, e, top_n=top_n
        )
        details[pd.Timestamp(eff)] = {"ranking": ranking, "locked": locked}
        row = {
            "EffectiveDate": pd.Timestamp(eff),
            "AsOfDate": pd.Timestamp(asof),
        }
        for i in range(top_n):
            row[f"Top{i+1}"] = selected[i] if i < len(selected) else None
        rows.append(row)

    schedule = pd.DataFrame(rows).sort_values("EffectiveDate").reset_index(drop=True)
    if schedule.empty:
        raise ValueError("Không tạo được selection schedule")
    return schedule, details

def selected_tickers_asof(selection_schedule, effective_date, top_n=DEFAULT_TOP_N):
    x = selection_schedule[selection_schedule["EffectiveDate"] <= pd.Timestamp(effective_date)]
    if x.empty:
        return []
    row = x.iloc[-1]
    return [row[f"Top{i+1}"] for i in range(top_n) if pd.notna(row.get(f"Top{i+1}"))]

# ==============================================================================
# 7. CÁC HỌ MÔ HÌNH PHÂN BỔ TRỌNG SỐ (EW, MPT-LW, MINVAR-LW, HRP, BLEND50)
# ==============================================================================
def close_matrix(price_dict, tickers, end=None, start=None):
    series = [price_dict[t]["Close"].rename(t) for t in tickers if t in price_dict]
    if not series:
        return pd.DataFrame()
    m = pd.concat(series, axis=1, join="inner").sort_index().dropna(how="any")
    if start is not None:
        m = m.loc[m.index >= pd.Timestamp(start)]
    if end is not None:
        m = m.loc[m.index <= pd.Timestamp(end)]
    return m

def equal_weights(tickers):
    tickers = list(tickers)
    return pd.Series(1 / len(tickers), index=tickers, dtype=float) if tickers else pd.Series(dtype=float)

def robust_mu(prices, lookback=DEFAULT_WEIGHT_LOOKBACK, mu_shrinkage=0.50):
    r = prices.pct_change(fill_method=None).dropna().tail(lookback)
    if r.empty:
        return pd.Series(0.0, index=prices.columns)
    span = max(20, min(lookback, len(r)))
    raw = r.ewm(span=span, adjust=False).mean().iloc[-1] * TRADING_DAYS
    grand = float(raw.mean())
    return mu_shrinkage * raw + (1 - mu_shrinkage) * grand

def lw_cov(prices, lookback=DEFAULT_WEIGHT_LOOKBACK):
    r = prices.pct_change(fill_method=None).dropna().tail(lookback)
    if len(r) < 30:
        return r.cov() * TRADING_DAYS
    est = LedoitWolf().fit(r.values)
    return pd.DataFrame(est.covariance_ * TRADING_DAYS, index=r.columns, columns=r.columns)

def max_sharpe_weights(mu, cov, max_weight=0.50, l2_gamma=0.03):
    tickers = list(mu.index)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    max_weight = max(max_weight, 1 / n)
    muv = mu.values.astype(float)
    cv = cov.reindex(index=tickers, columns=tickers).fillna(0).values.astype(float) + np.eye(n) * 1e-8

    def obj(w):
        ret = float(w @ muv)
        vol = math.sqrt(max(float(w @ cv @ w), 1e-12))
        return -(ret / vol) + l2_gamma * float(np.sum(w * w))

    x0 = np.repeat(1 / n, n)
    bounds = [(0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    if not res.success or np.any(~np.isfinite(res.x)):
        return equal_weights(tickers)
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=tickers) if w.sum() > 0 else equal_weights(tickers)

def min_variance_weights(cov, max_weight=0.50):
    tickers = list(cov.index)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    max_weight = max(max_weight, 1 / n)
    cv = cov.values.astype(float) + np.eye(n) * 1e-8

    def obj(w):
        return float(w @ cv @ w)

    x0 = np.repeat(1 / n, n)
    bounds = [(0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        return equal_weights(tickers)
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=tickers)

def _cluster_var(cov, items):
    sub = cov.loc[items, items].values
    ivp = 1 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)

def _quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = int(link[-1, 3])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix.loc[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def hrp_weights(prices, method="single"):
    r = prices.pct_change(fill_method=None).dropna()
    if len(r) < 30 or prices.shape[1] < 2:
        return equal_weights(prices.columns)
    cov = r.cov() * TRADING_DAYS
    corr = r.corr().clip(-1, 1)
    dist = np.sqrt((1 - corr) / 2)
    link = linkage(squareform(dist.values, checks=False), method=method)
    ordered = corr.index[_quasi_diag(link)].tolist()

    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            s = len(c) // 2
            c0, c1 = c[:s], c[s:]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            a = 1 - v0 / (v0 + v1) if v0 + v1 > 0 else 0.5
            w[c0] *= a
            w[c1] *= (1 - a)
            nxt += [c0, c1]
        clusters = nxt
    return w.reindex(prices.columns).fillna(0) / w.sum()

def compute_weights(model_family, prices):
    tickers = list(prices.columns)
    ew = equal_weights(tickers)
    if len(tickers) == 0 or model_family == "EW":
        return ew
    cov = lw_cov(prices, DEFAULT_WEIGHT_LOOKBACK)
    if model_family == "MINVAR_LW":
        return min_variance_weights(cov)
    if model_family == "HRP":
        return hrp_weights(prices.tail(DEFAULT_WEIGHT_LOOKBACK))

    mu = robust_mu(prices, DEFAULT_WEIGHT_LOOKBACK)
    mpt = max_sharpe_weights(mu, cov)
    if model_family == "MPT_LW":
        return mpt
    if model_family == "BLEND50":
        w = 0.50 * mpt + 0.50 * ew
        return w / w.sum()
    return ew

def build_dynamic_weight_schedule(price_dict, selection_schedule, calendar, start, end, model_family, allocation_freq=DEFAULT_ALLOCATION_FREQ, top_n=DEFAULT_TOP_N):
    rebalance_dates = first_session_each_period(calendar, start, end, allocation_freq)
    rows = []
    all_tickers = set()

    for eff in rebalance_dates:
        asof = previous_session(calendar, eff)
        if asof is None:
            continue
        top = selected_tickers_asof(selection_schedule, eff, top_n)
        if not top:
            continue
        all_tickers.update(top)
        prices = close_matrix(price_dict, top, end=asof)
        if prices.empty or len(prices) < 30:
            w = equal_weights(top)
        else:
            w = compute_weights(model_family, prices.tail(DEFAULT_WEIGHT_LOOKBACK + 1))
        row = {"EffectiveDate": pd.Timestamp(eff), "AsOfDate": pd.Timestamp(asof)}
        row.update({t: float(w.get(t, 0.0)) for t in top})
        rows.append(row)

    if not rows:
        raise ValueError("Không tạo được weight schedule")

    meta = pd.DataFrame(rows).sort_values("EffectiveDate").reset_index(drop=True)
    tickers = sorted(all_tickers)
    weights = meta.set_index("EffectiveDate").reindex(columns=tickers).fillna(0.0)
    return weights, meta[["EffectiveDate", "AsOfDate"]]

def build_static_buyhold_schedule(price_dict, selection_schedule, calendar, start, end, top_n=DEFAULT_TOP_N):
    start = pd.Timestamp(start)
    top = selected_tickers_asof(selection_schedule, start, top_n)
    if not top:
        raise ValueError("Không có Top N ở đầu kỳ cho BuyHold")
    return pd.DataFrame([equal_weights(top)], index=[start])

# ==============================================================================
# 8. MARKET REGIME & EVENT-DRIVEN PORTFOLIO ENGINE
# ==============================================================================
def build_market_regime(benchmark_features, span=200, slope_lookback=20, use_slope=True):
    x = benchmark_features.copy()
    ema_m = x["Close"].ewm(span=span, adjust=False, min_periods=span).mean()
    slope = ema_m - ema_m.shift(slope_lookback)
    risk_close = x["Close"] > ema_m
    if use_slope:
        risk_close = risk_close & (slope > 0)
    return pd.DataFrame({
        "EMA_Market": ema_m,
        "EMA_Slope": slope,
        "RiskOn_Close": risk_close.fillna(False),
        "RiskOn_Open": risk_close.shift(1).fillna(False),
    }, index=x.index)

def build_stock_open_masks(feature_store, tickers, b, e):
    masks = {}
    for t in tickers:
        if t in feature_store:
            active_close = stateful_active_from_score(feature_store[t]["TA_Score"], b, e)
            masks[t] = active_close.shift(1).fillna(0).astype(bool)
    return pd.DataFrame(masks).sort_index()

def run_portfolio_engine_dynamic(price_dict, weight_schedule, calendar, start, end,
                                 market_regime_open=None, stock_active_open=None,
                                 use_market_regime=True, use_stock_mask=True,
                                 individual_trailing_stop=False, individual_stop_pct=0.12,
                                 portfolio_trailing_stop=False, portfolio_stop_pct=0.15,
                                 initial_cash=DEFAULT_INITIAL_CASH, commission=DEFAULT_COMMISSION,
                                 slippage=DEFAULT_SLIPPAGE, fractional=False):
    tickers = list(weight_schedule.columns)
    idx = pd.DatetimeIndex(calendar)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(idx) == 0:
        raise ValueError("Không có ngày giao dịch trong kỳ")

    opens = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    closes = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        if t in price_dict:
            opens[t] = price_dict[t]["Open"].reindex(idx)
            closes[t] = price_dict[t]["Close"].reindex(idx)

    ws = weight_schedule.sort_index().reindex(columns=tickers)
    ws = ws.reindex(ws.index.union(idx)).sort_index().ffill().reindex(idx).fillna(0.0)

    regime = (pd.Series(True, index=idx) if market_regime_open is None
              else pd.Series(market_regime_open).reindex(idx).ffill().fillna(False).astype(bool))
    masks = (pd.DataFrame(True, index=idx, columns=tickers) if stock_active_open is None
             else stock_active_open.reindex(index=idx, columns=tickers).fillna(False).astype(bool))

    cash = float(initial_cash)
    shares = pd.Series(0.0, index=tickers)
    last_close = pd.Series(np.nan, index=tickers)
    last_target = pd.Series(np.nan, index=tickers)
    entry_peak = pd.Series(np.nan, index=tickers)
    stop_blocked = pd.Series(False, index=tickers)
    portfolio_peak = initial_cash
    portfolio_blocked = False

    orders, eq_rows, exposure_rows = [], [], []
    turnover, rebals = 0.0, 0

    for dt in idx:
        op = opens.loc[dt].astype(float)
        cl = closes.loc[dt].astype(float)

        valid_close = cl.notna()
        last_close.loc[valid_close] = cl.loc[valid_close]
        mark_open = op.where(op.notna(), last_close)
        mark_close = cl.where(cl.notna(), last_close)

        base = ws.loc[dt].clip(lower=0)
        if base.sum() > 1 + 1e-8:
            base = base / base.sum()

        risk_on = bool(regime.loc[dt]) if use_market_regime else True
        active = masks.loc[dt].astype(bool) if use_stock_mask else pd.Series(True, index=tickers)
        for t in tickers:
            if not active.get(t, False):
                stop_blocked[t] = False

        target = base.copy()
        if use_stock_mask:
            target *= active.astype(float)
        if individual_trailing_stop:
            target[stop_blocked] = 0.0
        if use_market_regime and not risk_on:
            target[:] = 0.0
        if portfolio_trailing_stop and portfolio_blocked:
            target[:] = 0.0

        target = target.clip(lower=0)
        if target.sum() > 1 + 1e-8:
            target /= target.sum()

        changed = last_target.isna().any() or (target - last_target.fillna(0)).abs().max() > 1e-10
        if changed:
            equity_open = cash + float((shares * mark_open.fillna(0)).sum())
            desired = shares.copy()

            for t in tickers:
                if pd.isna(op[t]) or op[t] <= 0:
                    continue
                if target[t] <= 0:
                    desired[t] = 0.0
                else:
                    px = op[t] * (1 + slippage)
                    raw_qty = equity_open * target[t] / (px * (1 + commission))
                    desired[t] = raw_qty if fractional else math.floor(raw_qty)

            # Bán trước
            for t in tickers:
                d = desired[t] - shares[t]
                if d < -1e-12 and pd.notna(op[t]) and op[t] > 0:
                    qty = -d
                    px = op[t] * (1 - slippage)
                    notional = qty * px
                    fee = notional * commission
                    cash += notional - fee
                    shares[t] -= qty
                    turnover += notional
                    orders.append({"Date": dt, "Ticker": t, "Side": "SELL", "Shares": qty, "ExecPrice": px, "Fee": fee, "Reason": "REBALANCE"})
                    if shares[t] <= 1e-12:
                        entry_peak[t] = np.nan

            # Mua sau
            for t in tickers:
                d = desired[t] - shares[t]
                if d > 1e-12 and pd.notna(op[t]) and op[t] > 0:
                    px = op[t] * (1 + slippage)
                    unit = px * (1 + commission)
                    qty = min(d, cash / unit) if fractional else min(d, math.floor(cash / unit))
                    if qty > 0:
                        notional = qty * px
                        fee = notional * commission
                        was_zero = shares[t] <= 1e-12
                        cash -= notional + fee
                        shares[t] += qty
                        turnover += notional
                        orders.append({"Date": dt, "Ticker": t, "Side": "BUY", "Shares": qty, "ExecPrice": px, "Fee": fee, "Reason": "REBALANCE"})
                        if was_zero and pd.notna(mark_close[t]):
                            entry_peak[t] = mark_close[t]

            last_target = target.copy()
            rebals += 1

        if individual_trailing_stop:
            for t in tickers:
                if shares[t] > 0 and pd.notna(mark_close[t]):
                    entry_peak[t] = max(entry_peak[t] if pd.notna(entry_peak[t]) else mark_close[t], mark_close[t])
                    if mark_close[t] <= entry_peak[t] * (1 - individual_stop_pct):
                        stop_blocked[t] = True

        equity = float(cash + (shares * mark_close.fillna(0)).sum())
        portfolio_peak = max(portfolio_peak, equity)
        if portfolio_trailing_stop and equity <= portfolio_peak * (1 - portfolio_stop_pct):
            portfolio_blocked = True

        invested = float((shares * mark_close.fillna(0)).sum())
        exposure = invested / equity if equity > 0 else 0.0
        exposure_rows.append(exposure)
        eq_rows.append((dt, equity))

    eq = pd.Series(dict(eq_rows)).sort_index()
    st_res = metrics_from_equity(eq)
    st_res.update({
        "#Orders": len(orders),
        "#Rebalances": rebals,
        "Turnover [%]": 100 * turnover / initial_cash,
        "AvgExposure [%]": 100 * np.mean(exposure_rows) if exposure_rows else 0.0,
        "AvgCash [%]": 100 * (1 - np.mean(exposure_rows)) if exposure_rows else 100.0,
    })
    return st_res, eq, pd.DataFrame(orders), pd.Series(exposure_rows, index=idx)

def benchmark_nav(benchmark_px, start, end, initial_cash=DEFAULT_INITIAL_CASH):
    x = benchmark_px.loc[(benchmark_px.index >= pd.Timestamp(start)) & (benchmark_px.index <= pd.Timestamp(end)), "Close"].dropna()
    if x.empty:
        return pd.Series(dtype=float)
    return initial_cash * x / x.iloc[0]

def build_synthetic_market_proxy(price_dict, exclude=None):
    exclude = set(exclude or [])
    closes = [px["Close"].rename(t) for t, px in price_dict.items() if t not in exclude]
    m = pd.concat(closes, axis=1).sort_index()
    ret = m.pct_change(fill_method=None)
    ewret = ret.mean(axis=1, skipna=True).fillna(0)
    close = 1000 * (1 + ewret).cumprod()
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0.0}, index=close.index)

# ==============================================================================
# 9. VALIDATION CHỌN CHAMPION & FINAL TEST ĐÁNH GIÁ
# ==============================================================================
def model_validation_score(stats, benchmark_stats):
    sh = stats.get("Sharpe", np.nan)
    sh = -3 if not np.isfinite(sh) else sh
    so = stats.get("Sortino", np.nan)
    so = -3 if not np.isfinite(so) else so
    ret = stats.get("Return [%]", -100)
    dd = stats.get("MaxDD [%]", -100)
    turnover = stats.get("Turnover [%]", 0)
    bret = benchmark_stats.get("Return [%]", 0)
    return 0.45 * sh + 0.15 * so + 0.012 * (ret - bret) + 0.004 * dd - 0.0005 * turnover

def evaluate_dynamic_models(price_dict, feature_store, benchmark_ticker, benchmark_features,
                            selection_schedule, calendar, start, end, b, e, top_n=DEFAULT_TOP_N,
                            model_specs=ACTIVE_MODEL_SPECS, use_market_regime=True, use_stock_mask=True):
    regime = build_market_regime(benchmark_features)
    union_tickers = sorted({
        selection_schedule.loc[i, f"Top{j+1}"]
        for i in selection_schedule.index for j in range(top_n)
        if pd.notna(selection_schedule.loc[i, f"Top{j+1}"])
    })
    masks = build_stock_open_masks(feature_store, union_tickers, b, e)

    bnav = benchmark_nav(price_dict[benchmark_ticker], start, end)
    bstats = metrics_from_equity(bnav)

    rows, curves, orders, schedules = [], {"BENCHMARK": bnav}, {}, {}
    for name, spec in model_specs.items():
        sched, _ = build_dynamic_weight_schedule(price_dict, selection_schedule, calendar, start, end, spec["family"], top_n=top_n)
        st_res, eq, od, _ = run_portfolio_engine_dynamic(
            price_dict, sched, calendar, start, end,
            market_regime_open=regime["RiskOn_Open"],
            stock_active_open=masks,
            use_market_regime=(use_market_regime and spec["regime"]),
            use_stock_mask=(use_stock_mask and spec["mask"]),
        )
        st_res["Model"] = name
        st_res["ModelName"] = spec["name"]
        st_res["ValidationScore"] = model_validation_score(st_res, bstats)
        rows.append(st_res)
        curves[name] = eq
        orders[name] = od
        schedules[name] = sched

    bstats.update({"Model": "BENCHMARK", "ModelName": "VN-Index / Market Benchmark", "ValidationScore": np.nan})
    rows.append(bstats)
    res = pd.DataFrame(rows).set_index("Model")
    champion = res.drop(index="BENCHMARK")["ValidationScore"].idxmax()
    return res, champion, curves, orders, schedules

def evaluate_final_test(price_dict, feature_store, benchmark_ticker, benchmark_features,
                        selection_schedule, calendar, start, end, b, e, champion, top_n=DEFAULT_TOP_N,
                        use_market_regime=True, use_stock_mask=True):
    regime = build_market_regime(benchmark_features)
    union_tickers = sorted({
        selection_schedule.loc[i, f"Top{j+1}"]
        for i in selection_schedule.index for j in range(top_n)
        if pd.notna(selection_schedule.loc[i, f"Top{j+1}"])
    })
    masks = build_stock_open_masks(feature_store, union_tickers, b, e)

    bnav = benchmark_nav(price_dict[benchmark_ticker], start, end)
    rows = []
    curves = {"VNINDEX_or_MarketBenchmark": bnav}
    order_map = {}
    exposure_map = {}

    bstats = metrics_from_equity(bnav)
    bstats["Model"] = "VNINDEX_or_MarketBenchmark"
    bstats["ModelName"] = "VN-Index / Benchmark"
    rows.append(bstats)

    bh_sched = build_static_buyhold_schedule(price_dict, selection_schedule, calendar, start, end, top_n=top_n)
    bh_stats, bh_eq, bh_od, bh_exp = run_portfolio_engine_dynamic(
        price_dict, bh_sched, calendar, start, end,
        market_regime_open=None, stock_active_open=None,
        use_market_regime=False, use_stock_mask=False
    )
    bh_stats["Model"] = "STATIC_EW_BH"
    bh_stats["ModelName"] = f"Static Buy & Hold (Top {top_n})"
    rows.append(bh_stats)
    curves["STATIC_EW_BH"] = bh_eq
    order_map["STATIC_EW_BH"] = bh_od
    exposure_map["STATIC_EW_BH"] = bh_exp

    ew_sched, _ = build_dynamic_weight_schedule(price_dict, selection_schedule, calendar, start, end, "EW", top_n=top_n)
    ew_stats, ew_eq, ew_od, ew_exp = run_portfolio_engine_dynamic(
        price_dict, ew_sched, calendar, start, end,
        market_regime_open=None, stock_active_open=None,
        use_market_regime=False, use_stock_mask=False
    )
    ew_stats["Model"] = "DYN_EW_NO_REGIME"
    ew_stats["ModelName"] = "Dynamic EW (No Regime)"
    rows.append(ew_stats)
    curves["DYN_EW_NO_REGIME"] = ew_eq
    order_map["DYN_EW_NO_REGIME"] = ew_od
    exposure_map["DYN_EW_NO_REGIME"] = ew_exp

    for name, spec in ACTIVE_MODEL_SPECS.items():
        sched, _ = build_dynamic_weight_schedule(price_dict, selection_schedule, calendar, start, end, spec["family"], top_n=top_n)
        st_res, eq, od, exp = run_portfolio_engine_dynamic(
            price_dict, sched, calendar, start, end,
            market_regime_open=regime["RiskOn_Open"],
            stock_active_open=masks,
            use_market_regime=use_market_regime,
            use_stock_mask=use_stock_mask
        )
        label = "LOCKED_CHAMPION" if name == champion else f"DIAG_{name}"
        st_res["Model"] = label
        st_res["ModelName"] = f"★ {spec['name']} (CHAMPION)" if name == champion else spec["name"]
        rows.append(st_res)
        curves[label] = eq
        order_map[label] = od
        exposure_map[label] = exp

    res = pd.DataFrame(rows).set_index("Model")
    bname = "VNINDEX_or_MarketBenchmark"
    res["ExcessCAGR_vs_Benchmark [pp]"] = res["CAGR [%]"] - res.loc[bname, "CAGR [%]"]
    res["DD_Improvement_vs_Benchmark [pp]"] = res["MaxDD [%]"] - res.loc[bname, "MaxDD [%]"]
    return res, curves, order_map, exposure_map

# ==============================================================================
# SIDEBAR - CẤU HÌNH & TẢI DỮ LIỆU
# ==============================================================================
st.sidebar.markdown("## ⚙️ Cấu Hình Chiến Lược")

st.sidebar.markdown("### 📁 Nguồn Dữ Liệu")
data_option = st.sidebar.radio(
    "Chọn nguồn file:",
    ["Tự động nhận diện trong thư mục", "Tải lên file CSV / Excel mới"],
    index=0
)

uploaded_file = None
if data_option == "Tải lên file CSV / Excel mới":
    uploaded_file = st.sidebar.file_uploader("Upload file OHLCV:", type=["csv", "xlsx", "xls"])

use_adjusted_prices = st.sidebar.checkbox("Sử dụng Giá điều chỉnh (Adjusted)", value=True)
top_n_param = st.sidebar.slider("Số cổ phiếu Top N:", min_value=3, max_value=15, value=5, step=1)
initial_cash_param = st.sidebar.number_input("Vốn ban đầu (VND):", min_value=10_000_000.0, value=DEFAULT_INITIAL_CASH, step=100_000_000.0, format="%.0f")
commission_pct = st.sidebar.number_input("Phí giao dịch (%):", min_value=0.0, max_value=1.0, value=0.15, step=0.01) / 100.0
slippage_pct = st.sidebar.number_input("Trượt giá Slippage (%):", min_value=0.0, max_value=1.0, value=0.05, step=0.01) / 100.0

st.sidebar.markdown("### 🛡️ Quản Trị Rủi Ro")
use_market_regime_param = st.sidebar.checkbox("Kích hoạt Market Regime (EMA200)", value=True)
use_stock_mask_param = st.sidebar.checkbox("Kích hoạt Stock TA Active Mask", value=True)

# Tìm file dữ liệu thông minh ở thư mục hiện tại hoặc thư mục cha
def find_local_data_file():
    search_dirs = [".", ".."]
    for d in search_dirs:
        if os.path.exists(d):
            patterns = [
                os.path.join(d, "HOSE*.csv"),
                os.path.join(d, "*HOSE*.csv"),
                os.path.join(d, "HOSE*.xlsx"),
                os.path.join(d, "*HOSE*.xlsx"),
                os.path.join(d, "*.csv"),
                os.path.join(d, "*.xlsx"),
            ]
            for pat in patterns:
                matches = glob.glob(pat)
                for f in matches:
                    fname = os.path.basename(f)
                    if not any(skip in fname.lower() for skip in ["quality", "split", "leakage", "output", "log"]):
                        if os.path.getsize(f) > 1000:
                            return f
    return None

raw_data = None
data_source_name = ""

if uploaded_file is not None:
    data_source_name = uploaded_file.name
    try:
        raw_data = load_data_file(uploaded_file)
    except Exception as e:
        st.error(f"Lỗi khi đọc file upload: {e}")
else:
    local_f = find_local_data_file()
    if local_f is not None and os.path.exists(local_f):
        data_source_name = local_f
        try:
            raw_data = load_data_file(local_f)
        except Exception as e:
            st.error(f"Lỗi khi đọc file cục bộ {local_f}: {e}")
    else:
        st.warning("Chưa tìm thấy file dữ liệu. Vui lòng upload file CSV hoặc XLSX của thị trường HOSE.")

# ==============================================================================
# TIÊU ĐỀ ỨNG DỤNG & HEADER BADGES
# ==============================================================================
st.markdown('<div class="main-title">📈 Hệ Thống Kiểm Định Định Lượng HOSE V4</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Kiến trúc Nested Walk-Forward | Lựa chọn Top N Causal | Tối ưu hóa phân bổ MPT & HRP | Triệt tiêu Future Leakage</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-container">
    <span class="tech-badge-primary">⚡ Nested Walk-Forward (Q / M)</span>
    <span class="tech-badge">📊 TA Score 0–10 (Group 1)</span>
    <span class="tech-badge-success">🛡️ Zero Future-Leakage</span>
    <span class="tech-badge">⚖️ Covariance Shrinkage (Ledoit–Wolf)</span>
    <span class="tech-badge">🎯 Hierarchical Risk Parity (HRP)</span>
    <span class="tech-badge">💵 Real Cash & Regime Overlay</span>
</div>
""", unsafe_allow_html=True)

if raw_data is None:
    st.info("👋 Hãy tải lên bộ dữ liệu OHLCV thị trường ở thanh bên trái hoặc đặt file dữ liệu vào thư mục dự án để bắt đầu.")
    st.stop()

# ==============================================================================
# PIPELINE XỬ LÝ TOÀN PHẦN CÓ CACHE
# ==============================================================================
@st.cache_data(show_spinner=False)
def process_full_quant_pipeline(df_raw, use_adj, top_n, initial_cash, comm, slip, use_regime, use_mask):
    clean_df = standardize_ohlcv(df_raw, use_adjusted=use_adj)
    q_table = data_quality_table(clean_df)

    dates, core_end, val_start, val_end, test_start, test_end = resolve_auto_splits(clean_df)
    s_table = pd.DataFrame([
        {"Tập dữ liệu": "Core Train", "Bắt đầu": dates[0].strftime("%Y-%m-%d"), "Kết thúc": core_end.strftime("%Y-%m-%d"), "Số phiên": int((dates <= core_end).sum()), "Nhiệm vụ": "Tối ưu hóa & Khóa cặp ngưỡng BUY/EXIT"},
        {"Tập dữ liệu": "Validation", "Bắt đầu": val_start.strftime("%Y-%m-%d"), "Kết thúc": val_end.strftime("%Y-%m-%d"), "Số phiên": int(((dates >= val_start) & (dates <= val_end)).sum()), "Nhiệm vụ": "Nested Walk-Forward & Đấu trường chọn Champion"},
        {"Tập dữ liệu": "Final Test", "Bắt đầu": test_start.strftime("%Y-%m-%d"), "Kết thúc": test_end.strftime("%Y-%m-%d"), "Số phiên": int((dates >= test_start).sum()), "Nhiệm vụ": "Kiểm định Out-Of-Sample độc lập của Champion đã khóa"},
    ])

    p_dict = {
        t: g[["Date", "Open", "High", "Low", "Close", "Volume"]].drop_duplicates("Date").sort_values("Date").set_index("Date")
        for t, g in clean_df.groupby("Ticker")
    }
    b_ticker = next((t for t in p_dict if t in BENCHMARK_ALIASES), None)
    if b_ticker is None:
        b_ticker = "__MARKET_PROXY__"
        p_dict[b_ticker] = build_synthetic_market_proxy(p_dict)

    f_store = {t: calculate_ta_features(px) for t, px in p_dict.items()}

    core_univ = build_universe(p_dict, core_end, b_ticker)
    core_cands = core_univ.loc[core_univ["Candidate"], "Ticker"].tolist()
    core_dates = dates[dates <= core_end]
    inner_n = max(63, min(126, int(0.30 * len(core_dates))))
    inner_start = core_dates[-inner_n]
    threshold_folds = make_equal_folds(core_dates, inner_start, core_end, 3)

    best_buy, best_exit, cv_trials = choose_global_ta_thresholds(f_store, core_cands, threshold_folds)

    val_sched, val_details = build_selection_schedule(
        f_store, p_dict, b_ticker, f_store[b_ticker], dates, val_start, val_end, best_buy, best_exit, top_n=top_n
    )
    val_results, champ, val_curves, val_orders, _ = evaluate_dynamic_models(
        p_dict, f_store, b_ticker, f_store[b_ticker], val_sched, dates, val_start, val_end, best_buy, best_exit,
        top_n=top_n, use_market_regime=use_regime, use_stock_mask=use_mask
    )

    test_sched, test_details = build_selection_schedule(
        f_store, p_dict, b_ticker, f_store[b_ticker], dates, test_start, test_end, best_buy, best_exit, top_n=top_n
    )
    test_results, test_curves, test_orders, test_exposures = evaluate_final_test(
        p_dict, f_store, b_ticker, f_store[b_ticker], test_sched, dates, test_start, test_end, best_buy, best_exit,
        champ, top_n=top_n, use_market_regime=use_regime, use_stock_mask=use_mask
    )

    return {
        "clean_df": clean_df,
        "q_table": q_table,
        "s_table": s_table,
        "dates": dates,
        "b_ticker": b_ticker,
        "p_dict": p_dict,
        "f_store": f_store,
        "best_buy": best_buy,
        "best_exit": best_exit,
        "cv_trials": cv_trials,
        "val_sched": val_sched,
        "val_details": val_details,
        "val_results": val_results,
        "champ": champ,
        "val_curves": val_curves,
        "test_sched": test_sched,
        "test_details": test_details,
        "test_results": test_results,
        "test_curves": test_curves,
        "test_orders": test_orders,
        "test_exposures": test_exposures
    }

with st.spinner("🔄 Đang xử lý tính toán định lượng V4 (Chuẩn hóa Encoding → TA Score → Core Train CV → Nested Selection → MPT/HRP → Final Test)..."):
    try:
        pipeline = process_full_quant_pipeline(
            raw_data, use_adjusted_prices, top_n_param, initial_cash_param,
            commission_pct, slippage_pct, use_market_regime_param, use_stock_mask_param
        )
    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {e}")
        st.stop()

# ==============================================================================
# CÁC TAB CHỨC NĂNG CHÍNH
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 1. Dữ Liệu & Phân Khúc",
    "⚙️ 2. Tối Ưu Ngưỡng Core Train",
    "🔄 3. Lịch Trình Top N",
    "🏆 4. Đấu Trường Validation",
    "🎯 5. Kiểm Định Final Test",
    "📋 6. Nhật Ký Lệnh & Trọng Số",
    "🔍 7. Phân Tích Mã Riêng Lẻ",
    "🛡️ 8. Audit Chống Leakage"
])

# ------------------------------------------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU & PHÂN KHÚC THỜI GIAN
# ------------------------------------------------------------------------------
with tab1:
    st.markdown("### 📊 Tổng Quan Bộ Dữ Liệu & Kiểm Tra Chất Lượng")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-title">Số Mã Cổ Phiếu</span><span>🏢</span></div>
            <div class="kpi-value">{pipeline['clean_df']['Ticker'].nunique():,}</div>
            <div class="kpi-sub">Universe HOSE ({data_source_name})</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-title">Tổng Phiên Giao Dịch</span><span>📅</span></div>
            <div class="kpi-value">{len(pipeline['dates']):,}</div>
            <div class="kpi-sub">{pipeline['dates'][0].strftime('%d/%m/%Y')} → {pipeline['dates'][-1].strftime('%d/%m/%Y')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-title">Nguồn Giá Sử Dụng</span><span>🏷️</span></div>
            <div class="kpi-value">{'Adjusted' if use_adjusted_prices else 'Raw'}</div>
            <div class="kpi-sub">Bảo toàn tỷ lệ trả cổ tức</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header"><span class="kpi-title">Benchmark Thị Trường</span><span>🎯</span></div>
            <div class="kpi-value">{pipeline['b_ticker']}</div>
            <div class="kpi-sub">Thước đo so sánh gốc</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### ⏳ Phân Chia Thời Gian Chống Future Leakage")
    st.dataframe(pipeline["s_table"], use_container_width=True, hide_index=True)

    st.markdown("#### 📋 Bảng Kiểm Tra Chất Lượng Dữ Liệu Từng Mã (Data Quality)")
    st.dataframe(pipeline["q_table"], use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2: TỐI ƯU HÓA NGƯỠNG TA TRÊN CORE TRAIN
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("### ⚙️ Tối Ưu Hóa & Khóa Ngưỡng BUY / EXIT Trên Core Train")
    st.markdown("""
    > **Nguyên tắc chống Look-ahead Bias:** Ngưỡng kích hoạt `BUY` và `EXIT` chỉ được tìm kiếm và khóa cố định trên tập **Core Train**.
    > Hàm mục tiêu kết hợp Median Sharpe, phạt độ lệch chuẩn (Stability Penalty) và hạn chế rủi ro sụt giảm vốn Max Drawdown.
    """)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"""
        <div class="status-box-champ">
            <h4 style="margin:0; color:#1E3A8A;">🟢 Ngưỡng BUY Đã Khóa: {pipeline['best_buy']} Điểm</h4>
            <p style="margin:4px 0 0 0; color:#475569; font-size:0.9rem;">Khi TA Score ≥ {pipeline['best_buy']} → Kích hoạt mở vị thế Mua.</p>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown(f"""
        <div class="status-box-warn">
            <h4 style="margin:0; color:#92400E;">🔴 Ngưỡng EXIT Đã Khóa: {pipeline['best_exit']} Điểm</h4>
            <p style="margin:4px 0 0 0; color:#78350F; font-size:0.9rem;">Khi TA Score ≤ {pipeline['best_exit']} → Thoát vị thế về Tiền mặt.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📊 Ma Trận Kết Quả Thử Nghiệm Cross-Validation")
    st.dataframe(
        pipeline["cv_trials"].style.format({
            "GlobalCVScore": "{:.4f}",
            "MedianSharpe": "{:.4f}",
            "SharpeStd": "{:.4f}",
            "MedianReturnPct": "{:.2f}%",
            "MedianMaxDDPct": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig_cv = px.bar(
        pipeline["cv_trials"],
        x=[f"B={r.BuyThreshold} | E={r.ExitThreshold}" for _, r in pipeline["cv_trials"].iterrows()],
        y="GlobalCVScore",
        color="GlobalCVScore",
        title="So Sánh Điểm Số Ổn Định Global CV Score Giữa Các Cặp Ngưỡng",
        labels={"x": "Cặp Ngưỡng (BUY | EXIT)", "GlobalCVScore": "Điểm CV Kết Hợp"},
        color_continuous_scale="Blues"
    )
    fig_cv.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_cv, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: LỊCH TRÌNH TÁI CHỌN TOP N (NESTED WALK-FORWARD)
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("### 🔄 Lịch Trình Tái Chọn Top N Cổ Phiếu Theo Quý")
    st.markdown("""
    > **Cơ chế Causal Point-in-time:** Danh mục Top N không bị đóng băng cứng nhắc cả năm. 
    > Tại mỗi ngày có hiệu lực `EffectiveDate`, Top N được chọn lại hoàn toàn bằng dữ liệu **chỉ đến phiên liền trước `AsOfDate (t-1)`**.
    """)

    sub_t1, sub_t2 = st.tabs(["🧪 Lịch Trình Validation", "🎯 Lịch Trình Final Test"])
    with sub_t1:
        st.dataframe(pipeline["val_sched"], use_container_width=True, hide_index=True)
    with sub_t2:
        st.dataframe(pipeline["test_sched"], use_container_width=True, hide_index=True)

    st.markdown("#### 🔍 Chi Tiết Bảng Điểm Xếp Hạng Tại Từng Kỳ Rebalance")
    test_eff_dates = [d.strftime("%Y-%m-%d") for d in pipeline["test_sched"]["EffectiveDate"]]
    selected_date_str = st.selectbox("Chọn mốc tái cân bằng Final Test để soi chi tiết:", test_eff_dates)

    if selected_date_str:
        sel_dt = pd.Timestamp(selected_date_str)
        if sel_dt in pipeline["test_details"]:
            detail_df = pipeline["test_details"][sel_dt]["locked"]
            cols_show = ["Ticker", "SelectedOrder", "FinalScore", "TA_Composite", "TrailingSharpe", "RelativeStrength126", "Vol63", "TA_Score", "RSI"]
            st.dataframe(
                detail_df[[c for c in cols_show if c in detail_df.columns]].style.format({
                    "FinalScore": "{:.2f}",
                    "TA_Composite": "{:.2f}",
                    "TrailingSharpe": "{:.3f}",
                    "RelativeStrength126": "{:.3f}",
                    "Vol63": "{:.3f}",
                    "RSI": "{:.1f}"
                }),
                use_container_width=True,
                hide_index=True
            )

# ------------------------------------------------------------------------------
# TAB 4: ĐẤU TRƯỜNG MÔ HÌNH TRÊN VALIDATION (CHỌN CHAMPION)
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("### 🏆 Đấu Trường Chọn Mô Hình Phân Bổ (Validation Set)")
    st.markdown("""
    > Tất cả 5 họ mô hình allocation (`EW`, `MPT_LW`, `MINVAR_LW`, `HRP`, `BLEND50`) cùng tranh tài trên tập Validation 
    > với **cùng một lịch Top N**. Mô hình đạt `ValidationScore` cao nhất được phong làm **LOCKED CHAMPION**.
    """)

    champ_name = pipeline["champ"]
    st.markdown(f"""
    <div class="status-box-champ">
        <h3 style="margin:0; color:#1E3A8A;">🥇 QUÁN QUÂN ĐÃ KHÓA: {ACTIVE_MODEL_SPECS.get(champ_name, {}).get('name', champ_name)}</h3>
        <p style="margin:6px 0 0 0; color:#334155; font-size:0.95rem;">
            Mô hình này chiến thắng xuất sắc trên Validation và được giữ nguyên để kiểm định Out-Of-Sample tại Final Test mà không được thay đổi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    val_res = pipeline["val_results"].copy()
    val_cols = ["ModelName", "Return [%]", "Sharpe", "Sortino", "MaxDD [%]", "Turnover [%]", "ValidationScore"]
    st.dataframe(
        val_res[[c for c in val_cols if c in val_res.columns]].sort_values("ValidationScore", ascending=False).style.format({
            "Return [%]": "{:.2f}%",
            "Sharpe": "{:.3f}",
            "Sortino": "{:.3f}",
            "MaxDD [%]": "{:.2f}%",
            "Turnover [%]": "{:.1f}%",
            "ValidationScore": "{:.4f}"
        }),
        use_container_width=True
    )

    fig_val = go.Figure()
    for m_key, m_curve in pipeline["val_curves"].items():
        name_display = ACTIVE_MODEL_SPECS.get(m_key, {}).get("name", m_key)
        fig_val.add_trace(go.Scatter(
            x=m_curve.index, y=m_curve.values,
            mode="lines", name=name_display,
            line=dict(width=3 if m_key == champ_name else 1.5)
        ))
    fig_val.update_layout(
        title="Đồ Thị Tăng Trưởng Tài Sản Trên Tập Validation (NAV Curves)",
        xaxis_title="Ngày", yaxis_title="NAV (VND)",
        template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_val, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: KIỂM ĐỊNH ĐỘC LẬP FINAL TEST (OUT-OF-SAMPLE)
# ------------------------------------------------------------------------------
with tab5:
    st.markdown("### 🎯 Kết Quả Kiểm Định Độc Lập Out-Of-Sample (Final Test)")
    st.markdown("""
    > **Kiểm định nghiêm ngặt:** Đánh giá Champion đã chọn trên tập dữ liệu hoàn toàn chưa từng thấy trong quá trình tối ưu.
    > So sánh trực tiếp với **Static Buy & Hold**, **Dynamic EW No Regime (Bóc tách Alpha)** và **Benchmark thị trường**.
    """)

    test_res = pipeline["test_results"]
    champ_row = test_res.loc["LOCKED_CHAMPION"] if "LOCKED_CHAMPION" in test_res.index else None

    if champ_row is not None:
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header"><span class="kpi-title">Total Return</span><span>📈</span></div>
                <div class="kpi-value" style="color:{'#16A34A' if champ_row['Return [%]'] >= 0 else '#DC2626'}">{champ_row['Return [%]']:.2f}%</div>
                <div class="kpi-sub">CAGR: {champ_row['CAGR [%]']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header"><span class="kpi-title">Sharpe Ratio</span><span>⚖️</span></div>
                <div class="kpi-value">{champ_row['Sharpe']:.3f}</div>
                <div class="kpi-sub">Sortino: {champ_row['Sortino']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header"><span class="kpi-title">Max Drawdown</span><span>🛡️</span></div>
                <div class="kpi-value" style="color:#DC2626">{champ_row['MaxDD [%]']:.2f}%</div>
                <div class="kpi-sub">Calmar: {champ_row['Calmar']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header"><span class="kpi-title">Excess CAGR</span><span>🚀</span></div>
                <div class="kpi-value" style="color:#1D4ED8">+{champ_row.get('ExcessCAGR_vs_Benchmark [pp]', 0):.2f}%</div>
                <div class="kpi-sub">Vượt trội so với Benchmark</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header"><span class="kpi-title">Giảm Max Drawdown</span><span>☔</span></div>
                <div class="kpi-value" style="color:#16A34A">+{champ_row.get('DD_Improvement_vs_Benchmark [pp]', 0):.2f}%</div>
                <div class="kpi-sub">Phòng thủ bảo vệ vốn</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 📊 Bảng Chỉ Số Hiệu Suất Toàn Diện (Final Test Performance)")
    cols_display = [
        "ModelName", "Return [%]", "CAGR [%]", "Sharpe", "Sortino", "MaxDD [%]", "Calmar",
        "Turnover [%]", "AvgExposure [%]", "AvgCash [%]",
        "ExcessCAGR_vs_Benchmark [pp]", "DD_Improvement_vs_Benchmark [pp]"
    ]
    st.dataframe(
        test_res[[c for c in cols_display if c in test_res.columns]].style.format({
            "Return [%]": "{:.2f}%",
            "CAGR [%]": "{:.2f}%",
            "Sharpe": "{:.3f}",
            "Sortino": "{:.3f}",
            "MaxDD [%]": "{:.2f}%",
            "Calmar": "{:.3f}",
            "Turnover [%]": "{:.1f}%",
            "AvgExposure [%]": "{:.1f}%",
            "AvgCash [%]": "{:.1f}%",
            "ExcessCAGR_vs_Benchmark [pp]": "{:+.2f}%",
            "DD_Improvement_vs_Benchmark [pp]": "{:+.2f}%"
        }),
        use_container_width=True
    )

    fig_test = go.Figure()
    key_models = ["LOCKED_CHAMPION", "STATIC_EW_BH", "DYN_EW_NO_REGIME", "VNINDEX_or_MarketBenchmark"]
    colors = {"LOCKED_CHAMPION": "#1D4ED8", "STATIC_EW_BH": "#F59E0B", "DYN_EW_NO_REGIME": "#8B5CF6", "VNINDEX_or_MarketBenchmark": "#64748B"}

    for km in key_models:
        if km in pipeline["test_curves"]:
            c_series = pipeline["test_curves"][km]
            m_label = test_res.loc[km, "ModelName"] if km in test_res.index else km
            fig_test.add_trace(go.Scatter(
                x=c_series.index, y=c_series.values,
                mode="lines", name=m_label,
                line=dict(color=colors.get(km, "#000000"), width=3 if km == "LOCKED_CHAMPION" else 1.8)
            ))
    fig_test.update_layout(
        title="Đường Cong Tăng Trưởng Tài Sản Out-Of-Sample (Final Test Equity Curves)",
        xaxis_title="Thời Gian", yaxis_title="NAV (VND)",
        template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_test, use_container_width=True)

    fig_dd = go.Figure()
    for km in key_models:
        if km in pipeline["test_curves"]:
            c_series = pipeline["test_curves"][km]
            dd_series = (c_series / c_series.cummax() - 1) * 100
            m_label = test_res.loc[km, "ModelName"] if km in test_res.index else km
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index, y=dd_series.values,
                mode="lines", name=m_label,
                line=dict(color=colors.get(km, "#000000"), width=2)
            ))
    fig_dd.update_layout(
        title="Đồ Thị Sụt Giảm Tài Sản Tối Đa (Underwater Drawdown Chart %)",
        xaxis_title="Thời Gian", yaxis_title="Drawdown (%)",
        template="plotly_white", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 6: NHẬT KÝ GIAO DỊCH & LỊCH SỬ TRỌNG SỐ
# ------------------------------------------------------------------------------
with tab6:
    st.markdown("### 📋 Nhật Ký Giao Dịch & Quản Lý Vị Thế")

    champ_orders = pipeline["test_orders"].get("LOCKED_CHAMPION", pd.DataFrame())
    if not champ_orders.empty:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**Tổng số lệnh khớp:** `{len(champ_orders):,}` lệnh giao dịch thực tế tại mức giá Open.")
        with c2:
            csv_buf = io.StringIO()
            champ_orders.to_csv(csv_buf, index=False)
            st.download_button("📥 Tải File CSV Nhật Ký Lệnh", data=csv_buf.getvalue(), file_name="HOSE_V4_Trade_Log.csv", mime="text/csv")

        st.dataframe(
            champ_orders.style.format({
                "Shares": "{:,.0f}",
                "ExecPrice": "{:,.2f}",
                "Fee": "{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Chưa có lệnh giao dịch nào được ghi nhận.")

    st.markdown("#### 📈 Tỷ Lệ Phân Bổ Cổ Phiếu vs Tiền Mặt Theo Thời Gian")
    if "LOCKED_CHAMPION" in pipeline["test_exposures"]:
        exp_series = pipeline["test_exposures"]["LOCKED_CHAMPION"] * 100
        cash_series = 100 - exp_series
        fig_alloc = go.Figure()
        fig_alloc.add_trace(go.Scatter(
            x=exp_series.index, y=exp_series.values,
            mode="lines", name="Đầu tư Cổ phiếu (%)",
            stackgroup="one", line=dict(width=0, color="#3B82F6")
        ))
        fig_alloc.add_trace(go.Scatter(
            x=cash_series.index, y=cash_series.values,
            mode="lines", name="Dự trữ Tiền mặt (%)",
            stackgroup="one", line=dict(width=0, color="#E2E8F0")
        ))
        fig_alloc.update_layout(
            title="Dòng Thời Gian Tỷ Trọng Phân Bổ Danh Mục (Equity Exposure vs Cash)",
            yaxis=dict(range=[0, 100], title="Tỷ Trọng (%)"),
            template="plotly_white", height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 7: PHÂN TÍCH KỸ THUẬT & TÍN HIỆU TỪNG MÃ RIÊNG LẺ
# ------------------------------------------------------------------------------
with tab7:
    st.markdown("### 🔍 Soi Tín Hiệu & Kỹ Thuật Chi Tiết Từng Cổ Phiếu")

    ticker_list = sorted([t for t in pipeline["p_dict"].keys() if t != pipeline["b_ticker"]])
    selected_stock = st.selectbox("Chọn cổ phiếu trong danh mục để xem phân tích:", ticker_list, index=0)

    if selected_stock in pipeline["f_store"]:
        stk_df = pipeline["f_store"][selected_stock]
        latest = stk_df.iloc[-1]
        active_close = stateful_active_from_score(stk_df["TA_Score"], pipeline["best_buy"], pipeline["best_exit"])
        cur_state = active_close.iloc[-1]

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header"><span class="kpi-title">Giá Phiên Gần Nhất</span><span>💰</span></div>
                <div class="kpi-value">{latest['Close']:.2f}</div>
                <div class="kpi-sub">Khối lượng: {latest['Volume']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header"><span class="kpi-title">Điểm TA Score</span><span>⭐</span></div>
                <div class="kpi-value">{int(latest['TA_Score'])} / 10</div>
                <div class="kpi-sub">Ngưỡng Mua: {pipeline['best_buy']} | Thoát: {pipeline['best_exit']}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header"><span class="kpi-title">Trạng Thái Vị Thế</span><span>📌</span></div>
                <div class="kpi-value" style="color:{'#16A34A' if cur_state == 1 else '#DC2626'}">{'MUA / NẮM GIỮ' if cur_state == 1 else 'TIỀN MẶT'}</div>
                <div class="kpi-sub">State: {cur_state}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header"><span class="kpi-title">Độ Biến Động (Vol63)</span><span>⚡</span></div>
                <div class="kpi-value">{latest['Vol63'] * 100:.1f}%</div>
                <div class="kpi-sub">RSI (14): {latest['RSI']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        fig_stock = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=[f"Biểu Đồ Kỹ Thuật {selected_stock} & Đường Trung Bình EMA", "Chỉ Báo Dòng Tiền OBV & MACD", "Điểm Đánh Giá Kỹ Thuật TA Score (0–10)"]
        )

        fig_stock.add_trace(go.Candlestick(
            x=stk_df.index, open=stk_df["Open"], high=stk_df["High"],
            low=stk_df["Low"], close=stk_df["Close"], name="OHLC"
        ), row=1, col=1)
        fig_stock.add_trace(go.Scatter(x=stk_df.index, y=stk_df["EMA20"], name="EMA20", line=dict(color="#3B82F6", width=1.5)), row=1, col=1)
        fig_stock.add_trace(go.Scatter(x=stk_df.index, y=stk_df["EMA50"], name="EMA50", line=dict(color="#F59E0B", width=1.5)), row=1, col=1)
        fig_stock.add_trace(go.Scatter(x=stk_df.index, y=stk_df["EMA200"], name="EMA200", line=dict(color="#DC2626", width=2)), row=1, col=1)

        fig_stock.add_trace(go.Scatter(x=stk_df.index, y=stk_df["OBV"], name="OBV", line=dict(color="#10B981", width=1.5)), row=2, col=1)
        fig_stock.add_trace(go.Scatter(x=stk_df.index, y=stk_df["OBV_EMA20"], name="OBV EMA20", line=dict(color="#6B7280", width=1.2, dash="dot")), row=2, col=1)

        fig_stock.add_trace(go.Scatter(x=stk_df.index, y=stk_df["TA_Score"], name="TA Score", line=dict(color="#8B5CF6", width=2)), row=3, col=1)
        fig_stock.add_hline(y=pipeline["best_buy"], line=dict(color="#16A34A", dash="dash"), row=3, col=1)
        fig_stock.add_hline(y=pipeline["best_exit"], line=dict(color="#DC2626", dash="dash"), row=3, col=1)

        fig_stock.update_layout(
            template="plotly_white", height=700,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_stock, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 8: AUDIT CHỐNG DATA LEAKAGE & PHƯƠNG PHÁP LUẬN
# ------------------------------------------------------------------------------
with tab8:
    st.markdown("### 🛡️ Ma Trận Kiểm Soát Rò Rỉ Thông Tin Tương Lai (Leakage Audit)")
    leakage_audit_df = pd.DataFrame([
        ["1. Ngưỡng TA BUY/EXIT", pipeline["s_table"].loc[0, "Kết thúc"], False, "Chỉ tối ưu và khóa cố định trên Core Train"],
        ["2. Họ Mô Hình (Model Family)", pipeline["s_table"].loc[1, "Kết thúc"], False, "Chỉ cạnh tranh và khóa trên Validation"],
        ["3. Tái chọn Top N hàng quý", "t-1", False, "Mỗi kỳ chỉ dùng dữ liệu lịch sử <= phiên trước"],
        ["4. Trọng số MPT/HRP hàng tháng", "t-1", False, "Ma trận hiệp phương sai chỉ dùng dữ liệu <= phiên trước"],
        ["5. Tín hiệu Market Regime", "t-1 Close", False, "Tín hiệu đóng cửa phiên t-1 chỉ giao dịch tại phiên t Open"],
        ["6. Thực thi giao dịch", "t Open", False, "Khớp lệnh tại giá mở cửa kèm Commission 0.15% & Slippage 0.05%"],
        ["7. Đánh giá Final Test", pipeline["s_table"].loc[2, "Kết thúc"], True, "Chỉ báo cáo kiểm định Out-Of-Sample; tuyệt đối không retune"]
    ], columns=["Hạng mục nghiên cứu", "Mốc thông tin tối đa", "Dùng Final Test?", "Quy tắc kiểm soát Leakage"])

    st.dataframe(leakage_audit_df, use_container_width=True, hide_index=True)

    st.markdown("""
    #### 📖 Tóm Tắt Phương Pháp Luận Định Lượng HOSE V4
    1. **Bài toán:** Từ Universe cổ phiếu lớn (97 mã), liên tục sàng lọc ra danh mục Top N cổ phiếu tối ưu và phân bổ tỷ trọng thông minh.
    2. **Khắc phục bẫy Overfitting:** Chia thời gian 3 đoạn tách biệt: Core Train → Validation → Final Test. Không sử dụng kết quả Test để chọn lại tham số.
    3. **Ưu điểm của Nested Walk-Forward:** Danh mục Top N thích nghi theo từng quý bằng dữ liệu causal point-in-time, giúp danh mục luôn nắm giữ các cổ phiếu dẫn dắt thị trường.
    4. **Quản trị rủi ro đa tầng:** Kết hợp bộ lọc trạng thái thị trường (Market Regime EMA200) và bộ lọc trạng thái kỹ thuật từng mã (Stock Mask) để chủ động đưa tài sản về Tiền mặt (Cash) khi thị trường bước vào xu hướng giảm (Downtrend / Bear Market).
    """)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
    Hệ Thống Kiểm Định Chiến Lược Định Lượng HOSE V4 © 2026 | Sẵn Sàng Triển Khai Trên Streamlit Community Cloud & GitHub
</div>
""", unsafe_allow_html=True)
