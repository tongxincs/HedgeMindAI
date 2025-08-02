from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.format import format_box
from datetime import datetime

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def strategist_agent() -> RunnableLambda:
    """
    Strategist Agent:
    Aggregates all agent reports and generates a professional investment outlook.

    Expects state with keys: 'symbol', 'fundamental_report', 'news_report', 'sentiment_report'

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
        print("🧩 Aggregating reports from fundamental, news, and sentiment agents...")

        # Get agent outputs
        fundamental = state["fundamental_report"]
        news = state["news_report"]
        sentiment = state["sentiment_report"]

        # Combine input for LLM
        combined_context = (
            f"Fundamental Report:\n{fundamental}\n\n"
            f"News Report:\n{news}\n\n"
            f"Reddit Sentiment Report:\n{sentiment}\n\n"
        )

        # Prompt for LLM
        prompt = (
            f"You are a senior investment strategist at a hedge fund.\n"
            f"Today is {today}. Your task is to review multiple analyses for stock {symbol}, "
            "and synthesize them into an actionable investment thesis.\n\n"
            f"{combined_context}\n\n"
            "Please provide:\n"
            "1. A brief summary of key insights from the fundamental, news, and Reddit sentiment reports.\n"
            "2. Your short-term view (next 1 month)\n"
            "3. Your medium-term view (1 to 12 months)\n"
            "4. Your long-term view (12+ months)\n"
            "5. Estimated directional bias over the next year (bullish, bearish, or range-bound), and why\n\n"
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
