import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from typing import Optional
import os
import glob


class FinanceVisualizer:
    """
    Generate financial charts for multi-modal analysis.
    Methods produce PNG files given a stock ticker and optional related data.
    """

    def __init__(self, outdir: str = "charts"):
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

    # --------------------------------------------------------
    # 1. Price trend + insider transactions overlay
    # --------------------------------------------------------
    def draw_price_with_insiders(
        self, symbol: str, insiders_df: Optional[pd.DataFrame] = None,
        period: str = "1y"
    ) -> str:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist.index, hist["Close"], label="Close Price", color="blue")

        if insiders_df is not None and not insiders_df.empty:
            buys = insiders_df[insiders_df["transaction"] == "Buy"]
            sells = insiders_df[insiders_df["transaction"] == "Sale"]
            ax.scatter(buys["startDate"], buys["price"], color="green", marker="^", s=60, label="Insider Buy")
            ax.scatter(sells["startDate"], sells["price"], color="red", marker="v", s=60, label="Insider Sale")

        ax.set_title(f"{symbol} Price with Insider Transactions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.autofmt_xdate()

        outfile = f"{self.outdir}/{symbol}_price_insiders.png"
        plt.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
        return outfile

    # --------------------------------------------------------
    # 2. Quarterly revenue vs. net income
    # --------------------------------------------------------
    def draw_quarterly_revenue_income(
        self, symbol: str, qdf: pd.DataFrame
    ) -> str:
        if qdf is None or qdf.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))
        quarters = [d.strftime("%Y-%m") for d in qdf.index]
        width = 0.35

        ax.bar([i - width/2 for i in range(len(qdf))], qdf["Revenue"], width, label="Revenue")
        ax.bar([i + width/2 for i in range(len(qdf))], qdf["NetIncome"], width, label="Net Income")

        ax.set_xticks(range(len(qdf)))
        ax.set_xticklabels(quarters, rotation=45)
        ax.set_title(f"{symbol} Quarterly Revenue vs Net Income")
        ax.set_ylabel("USD")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        outfile = f"{self.outdir}/{symbol}_quarterly_revenue_income.png"
        plt.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
        return outfile

    # --------------------------------------------------------
    # 3. Relative performance vs benchmark
    # --------------------------------------------------------
    def draw_relative_performance(
        self, symbol: str, benchmark: str = "SPY", period: str = "1y"
    ) -> str:
        stock = yf.Ticker(symbol)
        bench = yf.Ticker(benchmark)
        hist_s = stock.history(period=period)["Close"]
        hist_b = bench.history(period=period)["Close"]

        # Normalize both to 100 at start
        norm_s = (hist_s / hist_s.iloc[0]) * 100
        norm_b = (hist_b / hist_b.iloc[0]) * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(norm_s.index, norm_s, label=symbol, color="blue")
        ax.plot(norm_b.index, norm_b, label=benchmark, color="gray", linestyle="--")

        ax.set_title(f"{symbol} vs {benchmark} (Relative Performance)")
        ax.set_ylabel("Indexed (100 = start)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.autofmt_xdate()

        outfile = f"{self.outdir}/{symbol}_relative_perf.png"
        plt.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
        return outfile

    # --------------------------------------------------------
    # Cleanup method
    # --------------------------------------------------------
    def clear_images(self, pattern: str = "*.png") -> None:
        """
        Remove generated chart images in the output directory.

        Args:
            pattern: glob pattern for files to delete (default: all PNGs)
        """
        files = glob.glob(os.path.join(self.outdir, pattern))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"⚠️ Failed to remove {f}: {e}")
        print(f"🗑️ Cleared {len(files)} image(s) from {self.outdir}")
