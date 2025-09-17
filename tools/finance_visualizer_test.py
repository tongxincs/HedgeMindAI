# test_finance_visualizer.py

import yfinance as yf
from tools.yahoo_finance import extract_insider_transactions, extract_quarterly_earnings
from tools.finance_visualizer import FinanceVisualizer


def test_generate_images(symbol: str = "PLTR"):
    """
    Generates all finance visualizations for a given symbol.
    Saves charts into charts/ and prints file paths for manual verification.
    """
    print("=" * 60)
    print(f"[TEST] Generating finance visualizations for {symbol}...")

    viz = FinanceVisualizer()

    stock = yf.Ticker(symbol)

    # 1. Price trend with insider markers
    insiders, _ = extract_insider_transactions(stock, last_n=30) or (None, None)
    print(insiders.head())
    print(insiders.columns)

    img1 = viz.draw_price_with_insiders(symbol, insiders)
    print(f"✅ Price + Insider chart saved: {img1}")

    # 2. Quarterly revenue vs net income
    qdf = extract_quarterly_earnings(stock, max_quarters=8)
    if not qdf.empty:
        img2 = viz.draw_quarterly_revenue_income(symbol, qdf)
        print(f"✅ Quarterly Revenue vs Net Income chart saved: {img2}")
    else:
        print("⚠️ No quarterly earnings data available for this ticker.")

    # 3. Relative performance vs SPY
    img3 = viz.draw_relative_performance(symbol, benchmark="SPY")
    print(f"✅ Relative Performance chart saved: {img3}")

    print("=" * 60)
    print("📝 Open the generated PNGs in the charts/ folder to verify manually.\n")


if __name__ == "__main__":
    test_generate_images("PLTR")
