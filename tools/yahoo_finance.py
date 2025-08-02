import yfinance as yf
from datetime import datetime


def get_fundamentals(symbol: str) -> dict:
    stock = yf.Ticker(symbol)
    info = stock.info
    price_trend = get_historical_prices(symbol)

    return {
        "symbol": symbol,
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