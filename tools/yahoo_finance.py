import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def get_fundamentals(symbol: str) -> dict:
    stock = yf.Ticker(symbol)
    info = stock.info
    price_trend = get_historical_prices(symbol)

    return {
        "symbol": symbol,
        "industry": info.get('industry'),
        "date": datetime.today().strftime("%Y-%m-%d"),
        "marketCap": info.get("marketCap"),
        "forwardPE": info.get("forwardPE"),
        "revenueGrowth": info.get("revenueGrowth"),
        "profitMargins": info.get("profitMargins"),
        "operatingCashflow": info.get("operatingCashflow"),
        "freeCashflow": info.get("freeCashflow"),
        "debtToEquity": info.get("debtToEquity"),
        "price_trend": price_trend
    }


def get_historical_prices(symbol: str, period: str = "1y") -> dict:
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    closing_prices = hist["Close"]

    return {
        "start_date": closing_prices.index[0].strftime("%Y-%m-%d"),
        "end_date": closing_prices.index[-1].strftime("%Y-%m-%d"),
        "start_price": round(closing_prices.iloc[0], 2),
        "end_price": round(closing_prices.iloc[-1], 2),
        "percent_change": round(((closing_prices.iloc[-1] - closing_prices.iloc[0]) / closing_prices.iloc[0]) * 100, 2)
    }


def extract_quarterly_earnings(stock: yf.Ticker, max_quarters: int = 8) -> pd.DataFrame:
    """
    Pull quarterly income statement (yfinance), return columns:
    [Revenue, NetIncome, EPS_Diluted] for last `max_quarters` periods if available.
    """
    # Try new API first, then legacy
    raw = None
    for attr in ("quarterly_income_stmt", "quarterly_financials"):
        raw = getattr(stock, attr, None)
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            break
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = _normalize_quarterly_df(raw)

    # Column name variants seen on Yahoo
    revenue_key = _pick_first(df.columns, ["Total Revenue", "TotalRevenue", "Revenues", "OperatingRevenue"])
    netinc_key = _pick_first(df.columns, ["Net Income", "NetIncome", "NetIncomeCommonStockholders"])
    eps_dil_key = _pick_first(df.columns, ["Diluted EPS", "DilutedEPS", "EPS Diluted"])
    dil_shrs_key = _pick_first(df.columns, ["Diluted Average Shares", "DilutedAverageShares",
                                            "WeightedAverageDilutedSharesOutstanding"])

    out = pd.DataFrame(index=df.index)
    if revenue_key is not None:
        out["Revenue"] = pd.to_numeric(df[revenue_key], errors="coerce")
    if netinc_key is not None:
        out["NetIncome"] = pd.to_numeric(df[netinc_key], errors="coerce")

    # Prefer reported diluted EPS; else derive = NetIncome / DilutedShares
    if eps_dil_key is not None:
        out["EPS_Diluted"] = pd.to_numeric(df[eps_dil_key], errors="coerce")
    elif netinc_key is not None and dil_shrs_key is not None:
        ni = pd.to_numeric(df[netinc_key], errors="coerce")
        sh = pd.to_numeric(df[dil_shrs_key], errors="coerce")
        out["EPS_Diluted"] = ni / sh

    out = out.dropna(how="all")
    if max_quarters:
        out = out.tail(max_quarters)
    return out


def extract_insider_transactions(stock: yf.Ticker, last_n: int = 10):
    try:
        info = stock.info or {}
        if str(info.get("quoteType")).upper() == "ETF":
            return None
    except Exception:
        pass

    raw = getattr(stock, "insider_transactions", None)
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return None

    df = normalize_insider_transactions_df_basic(raw)
    if df.empty:
        return None

    summary = {
        "by_type_counts": df["transaction"].astype(str).value_counts(dropna=False).to_dict(),
        "net_shares": pd.to_numeric(df["signed_shares"], errors="coerce").sum(),
        "total_value": pd.to_numeric(df["value"], errors="coerce").sum(),
    }
    return df.tail(last_n), summary


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _iso(d) -> Optional[str]:
    if pd.isna(d) or d is None:
        return None
    if isinstance(d, (pd.Timestamp, )):
        return d.date().isoformat()
    if hasattr(d, "isoformat"):
        return d.isoformat()
    # try parse
    try:
        return pd.to_datetime(d).date().isoformat()
    except Exception:
        return None


def _standardize_tx_type(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    if "buy" in s or "purchase" in s or "acq" in s:
        return "Buy"
    if "sale" in s or "sell" in s or "dispos" in s:
        return "Sale"
    if "option" in s and ("exerc" in s or "conversion" in s):
        return "Option Exercise"
    if "gift" in s:
        return "Gift"
    if "award" in s or "grant" in s:
        return "Award/Grant"
    return s.title()


def normalize_insider_transactions_df_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, robust normalizer for yfinance insider_transactions.
    No regex, no text parsing — just renaming, types, price=value/abs(shares).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=[
            "startDate","filer","transaction","shares","value","price",
            "ownership","position","url","text","signed_shares"
        ])

    d = df.copy()

    # Map uppercase/spacey -> canonical lowercase keys commonly seen
    rename_map = {
        "Start Date": "startDate",
        "Insider": "filer",
        "Transaction": "transaction",
        "Shares": "shares",
        "Value": "value",
        "Ownership": "ownership",
        "Position": "position",
        "URL": "url",
        "Text": "text",
    }
    for old, new in rename_map.items():
        if old in d.columns and new not in d.columns:
            d = d.rename(columns={old: new})

    # Dates
    if "startDate" in d.columns:
        if pd.api.types.is_numeric_dtype(d["startDate"]):
            d["startDate"] = pd.to_datetime(d["startDate"], unit="s", utc=True).dt.tz_convert(None)
        else:
            d["startDate"] = pd.to_datetime(d["startDate"], errors="coerce")

    # Numerics
    if "shares" in d.columns:
        d["shares"] = pd.to_numeric(d["shares"], errors="coerce")
    if "value" in d.columns:
        d["value"] = pd.to_numeric(d["value"], errors="coerce")

    # Price: derive from value / abs(shares) when both exist (avoid negative prices)
    d["price"] = np.nan
    if "value" in d.columns and "shares" in d.columns:
        denom = d["shares"].replace(0, np.nan).abs()
        with np.errstate(divide="ignore", invalid="ignore"):
            d.loc[denom.notna() & d["value"].notna(), "price"] = d["value"] / denom

    # Transaction type smoothing (optional but useful)
    if "transaction" in d.columns:
        d["transaction"] = d["transaction"].astype(str).map(_standardize_tx_type)

    # Fallback: if transaction is blank, parse from text
    if (d["transaction"] == "").all() and "text" in d.columns:
        d["transaction"] = d["text"].map(_extract_tx_type_from_text)

    # signed_shares: Buys positive, Sales negative (only flip if positive)
    d["signed_shares"] = pd.to_numeric(d.get("shares"), errors="coerce")
    mask_sale = (d.get("transaction") == "Sale") & d["signed_shares"].gt(0)
    d.loc[mask_sale, "signed_shares"] = -d.loc[mask_sale, "signed_shares"]

    keep = ["startDate","filer","transaction","shares","value","price",
            "ownership","position","url","text","signed_shares"]
    for k in keep:
        if k not in d.columns:
            d[k] = None

    return d[keep].sort_values("startDate", na_position="last")


def _add_growth_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Add QoQ (shift=1) and YoY (shift=4) growth for each metric in `cols`.
    Uses pct_change (no deprecated options). Any division-by-zero produces
    ±inf which we convert to NA explicitly.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.sort_index()

    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")

        # QoQ and YoY growth
        qoq = s.pct_change(periods=1)
        yoy = s.pct_change(periods=4)

        # Replace ±inf from divide-by-zero with NA (no deprecated option_context)
        qoq = qoq.replace([np.inf, -np.inf], pd.NA)
        yoy = yoy.replace([np.inf, -np.inf], pd.NA)

        out[f"{c}_qoq"] = qoq
        out[f"{c}_yoy"] = yoy

    return out


def get_quarterly_earnings_json(symbol: str, max_quarters: int = 8) -> Dict[str, Any]:
    """
    Returns normalized last `max_quarters` (ascending) with QoQ/YoY growth.
    Schema:
      {
        "symbol": ...,
        "as_of": "YYYY-MM-DD",
        "quarters": [
           {"period": "YYYY-MM-DD", "revenue": float, "net_income": float, "eps_diluted": float,
            "growth": {"revenue":{"qoq":float|null,"yoy":float|null}, ...}}
        ],
      }
    """
    stock = yf.Ticker(symbol)
    qdf = extract_quarterly_earnings(stock, max_quarters=max_quarters)
    if qdf.empty:
        return {
            "symbol": symbol,
            "as_of": datetime.today().strftime("%Y-%m-%d"),
            "quarters": [],
            "note": "No quarterly statement data available from Yahoo."
        }

    # Ensure numeric
    for c in ["Revenue", "NetIncome", "EPS_Diluted"]:
        if c in qdf.columns:
            qdf[c] = pd.to_numeric(qdf[c], errors="coerce")

    # Add growth columns
    qdf = _add_growth_cols(qdf, ["Revenue", "NetIncome", "EPS_Diluted"])

    # Keep last N (already done inside extract), ensure ascending order
    qdf = qdf.sort_index()

    quarters = []
    for dt, row in qdf.iterrows():
        period = _iso(dt)
        if period is None:
            continue
        quarters.append({
            "period": period,
            "revenue": _to_float(row.get("Revenue")),
            "net_income": _to_float(row.get("NetIncome")),
            "eps_diluted": _to_float(row.get("EPS_Diluted")),
            "growth": {
                "revenue": {"qoq": _to_float(row.get("Revenue_qoq")), "yoy": _to_float(row.get("Revenue_yoy"))},
                "net_income": {"qoq": _to_float(row.get("NetIncome_qoq")), "yoy": _to_float(row.get("NetIncome_yoy"))},
                "eps_diluted": {"qoq": _to_float(row.get("EPS_Diluted_qoq")), "yoy": _to_float(row.get("EPS_Diluted_yoy"))},
            }
        })

    return {
        "symbol": symbol,
        "as_of": datetime.today().strftime("%Y-%m-%d"),
        "quarters": quarters,
    }


def get_insider_transactions_json(symbol: str, last_n: int = 20) -> Dict[str, Any]:
    """
    Returns normalized insider transactions with a compact summary.
    Schema:
      {
        "symbol": ...,
        "as_of": "YYYY-MM-DD",
        "window": {"start":"YYYY-MM-DD","end":"YYYY-MM-DD"},
        "summary": {"total_transactions": int, "by_type_counts": {...}, "net_shares": float|null, "total_value_usd": float|null},
        "transactions": [{"date": "...", "filer": str|null, "transaction": str|null, "ownership": str|null,
                          "shares": float|null, "price": float|null, "value": float|null}, ...],
      }
    """
    stock = yf.Ticker(symbol)
    ext = extract_insider_transactions(stock, last_n=last_n)
    as_of = datetime.today().strftime("%Y-%m-%d")

    if ext is None:
        return {
            "symbol": symbol,
            "as_of": as_of,
            "window": {"start": None, "end": None},
            "summary": {"total_transactions": 0, "by_type_counts": {}, "net_shares": None, "total_value_usd": None},
            "transactions": [],
            "note": "No insider transaction table available."
        }

    df_tail, summary = ext
    df_tail = df_tail.copy()

    # Normalize numeric fields & compute price if missing
    if "shares" in df_tail.columns:
        df_tail["shares"] = pd.to_numeric(df_tail["shares"], errors="coerce")
    if "value" in df_tail.columns:
        df_tail["value"] = pd.to_numeric(df_tail["value"], errors="coerce")
    if "price" not in df_tail.columns:
        # derive price if possible
        df_tail["price"] = None
        try:
            can = df_tail[["value", "shares"]].dropna()
            df_tail.loc[can.index, "price"] = can["value"] / can["shares"].replace(0, pd.NA)
        except Exception:
            pass

    # Window
    start = None
    end = None
    if "startDate" in df_tail.columns:
        good_dates = df_tail["startDate"].dropna()
        if not good_dates.empty:
            start = _iso(good_dates.min())
            end = _iso(good_dates.max())

    # Build transactions list
    cols = df_tail.columns
    records = []
    for _, r in df_tail.iterrows():
        records.append({
            "date": _iso(r.get("startDate")),
            "filer": (r.get("filer") if "filer" in cols else None),
            "transaction": (r.get("transaction") if "transaction" in cols else None),
            "ownership": (r.get("ownership") if "ownership" in cols else None),
            "shares": _to_float(r.get("shares")),
            "price": _to_float(r.get("price")),
            "value": _to_float(r.get("value")),
        })

    # Summary defaults
    by_type = summary.get("by_type_counts", {}) if isinstance(summary, dict) else {}
    net_shares = summary.get("net_shares") if isinstance(summary, dict) else None
    total_value = summary.get("total_value") if isinstance(summary, dict) else None

    return {
        "symbol": symbol,
        "as_of": as_of,
        "window": {"start": start, "end": end},
        "summary": {
            "total_transactions": len(records),
            "by_type_counts": by_type,
            "net_shares": _to_float(net_shares),
            "total_value_usd": _to_float(total_value),
        },
        "transactions": records,
    }


def _pick_first(colnames, candidates):
    for k in candidates:
        if k in colnames:
            return k
    return None


def _normalize_quarterly_df(rawdf: pd.DataFrame) -> pd.DataFrame:
    """Ensure rows=periods, cols=metrics; sort by period ascending."""
    if rawdf is None or rawdf.empty:
        return pd.DataFrame()

    df = rawdf.copy()

    # If metrics are rows and dates are columns, transpose.
    looks_like_metrics_on_index = False
    if isinstance(df.index, pd.Index):
        idx_vals = [str(x) for x in df.index.tolist()[:10]]
        looks_like_metrics_on_index = any(
            any(k in s for k in ["Revenue", "Income", "EPS", "Profit"]) for s in idx_vals
        )
    if looks_like_metrics_on_index:
        df = df.T

    # Index => datetime (quarter end)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    return df


def _extract_tx_type_from_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    if "sale" in t or "sell" in t or "dispos" in t:
        return "Sale"
    if "buy" in t or "purchase" in t or "acq" in t:
        return "Buy"
    if "option" in t and ("exerc" in t or "conversion" in t):
        return "Option Exercise"
    if "gift" in t:
        return "Gift"
    if "award" in t or "grant" in t:
        return "Award/Grant"
    return ""
