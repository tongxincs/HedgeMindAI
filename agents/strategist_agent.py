from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.format import format_box
from datetime import datetime

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def strategist_agent() -> RunnableLambda:
    """
    Strategist Agent:
    Aggregates all agent reports (fundamental, news, sentiment, OPTIONAL satellite)
    and generates a professional investment outlook.

    Expects state with keys:
      - 'symbol' (str)
      - 'fundamental_report' (str)
      - 'news_report' (str)
      - 'sentiment_report' (str)
      - OPTIONAL 'satellite_report' (JSON string produced by satellite module)

    Returns:
        {
            "symbol": ...,
            "investment_thesis": formatted string summary
        }
    """
    def _invoke(state: dict) -> dict:
        symbol = state["symbol"]
        today = datetime.today().strftime("%Y-%m-%d")

        print("-" * 60)
        print(f"📊 [Strategist Agent] Synthesizing insights for {symbol}...")
        print("🧩 Aggregating reports from fundamental, news, sentiment, and satellite...")

        # Collect agent outputs
        fundamental = state.get("fundamental_report", "")
        earnings = state.get("earnings_report", "")
        insider = state.get("insider_report", "")
        news = state.get("news_report", "")
        sentiment = state.get("sentiment_report", "")
        satellite = state.get("satellite_report", "")

        # Combine into context
        combined_context = (
            f"Fundamental Report:\n{fundamental}\n\n"
            f"Earnings:\n{earnings}\n\n"
            f"Insider Transactions Report:\n{insider}\n\n"
            f"News Report:\n{news}\n\n"
            f"Reddit Sentiment Report:\n{sentiment}\n\n"
            f"Satellite Report:\n{satellite}\n\n"
        )

        prompt = (
            f"You are a senior investment strategist at a hedge fund.\n"
            f"Today is {today}. Your task is to review multiple analyses for stock {symbol}, "
            "and synthesize them into an actionable investment thesis.\n\n"
            f"{combined_context}\n\n"
            "Please provide:\n"
            "1. A brief synthesis of key insights across all reports.\n"
            "2. Short-term view (1 month)\n"
            "3. Medium-term view (1–12 months)\n"
            "4. Long-term view (12+ months)\n"
            "5. Estimated directional bias for the next year (bullish, bearish, range-bound), with rationale\n\n"
            "Respond in a clean, professional format using numbered bullet points."
        )

        print("🤖 [Strategist Agent] Calling LLM for final investment outlook...")
        thesis = llm.invoke(prompt).content.strip()

        # Header box
        report_header = format_box([
            f"📈 Investment Strategy Report for {symbol}",
            f"🗓️ Date: {today}"
        ], width=90)

        # Full formatted report
        full_report = f"{report_header}\n{thesis}\n"

        print("✅ [Strategist Agent] Final report generated:\n")
        print(full_report)

        return {
            "symbol": symbol,
            "investment_thesis": full_report
        }

    return RunnableLambda(_invoke)
