import base64
import yfinance as yf
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.format import format_box
from tools.yahoo_finance import extract_insider_transactions, extract_quarterly_earnings
from tools.finance_visualizer import FinanceVisualizer

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _encode_image(path: str) -> str:
    """Encode local image file to base64 data URI string for Gemini."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def chart_agent() -> RunnableLambda:
    """
    Chart Analysis Agent:
    Generates financial charts (price+insiders, revenue vs income, relative performance),
    sends images + text context to Gemini for analysis,
    then deletes images.
    """

    def _invoke(state: dict) -> dict:
        symbol = state["symbol"]
        print("-" * 60)
        print(f"🖼️ [Chart Agent] Generating charts for {symbol}...")

        viz = FinanceVisualizer()
        stock = yf.Ticker(symbol)

        # 1. Insider overlay chart
        insiders, _ = extract_insider_transactions(stock, last_n=30) or (None, None)
        img1 = viz.draw_price_with_insiders(symbol, insiders)

        # 2. Quarterly revenue vs net income
        qdf = extract_quarterly_earnings(stock, max_quarters=8)
        img2 = viz.draw_quarterly_revenue_income(symbol, qdf)

        # 3. Relative performance vs SPY
        img3 = viz.draw_relative_performance(symbol, benchmark="SPY")

        print("📈 [Chart Agent] Charts generated successfully.")

        # Build multimodal input
        img_summary = (
            f"Generated 3 charts for {symbol}:\n"
            f"1. Price trend with insider buy/sell markers.\n"
            f"2. Quarterly revenue vs. net income (last 8 quarters).\n"
            f"3. Relative performance compared to SPY (last 1 year).\n"
            "Please analyze these visuals in a 3–5 point summary. "
            "Focus on insider timing, financial momentum, and market performance."
        )

        # Attach text + images
        message = HumanMessage(
            content=[
                {"type": "text", "text": img_summary},
                {"type": "image_url", "image_url": {"url": _encode_image(img1)}},
                {"type": "image_url", "image_url": {"url": _encode_image(img2)}},
                {"type": "image_url", "image_url": {"url": _encode_image(img3)}},
            ]
        )

        print("🤖 [Chart Agent] Sending charts to Gemini 2.5 Flash for analysis...")
        insights = llm.invoke([message]).content.strip()
        print("✅ [Chart Agent] Analysis complete.")

        # Cleanup images
        viz.clear_images(pattern=f"{symbol}_*.png")
        print("🗑️ [Chart Agent] Temporary chart images deleted.")
        print("-" * 60 + "\n")

        # Format report
        report_header = format_box([
            f"🖼️ Chart Analysis Report for {symbol}",
            f"📅 Date: today"
        ], width=90)

        full_report = f"{report_header}\n{insights}\n"
        print(full_report)

        return {
            "symbol": symbol,
            "chart_report": full_report
        }

    return RunnableLambda(_invoke)
