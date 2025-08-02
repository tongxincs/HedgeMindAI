from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.yahoo_finance import get_fundamentals
from tools.format import format_box

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def fundamental_agent() -> RunnableLambda:
    """
    Fundamental Analysis Agent:
    Fetches financial metrics and 1-year price trend for a stock ticker,
    then uses LLM to generate investor-focused insights.

    Returns:
        RunnableLambda accepting a state dict with "symbol", returns:
        {
            "symbol": ...,
            "fundamental_report": formatted summary string
        }
    """

    def _invoke(state: dict) -> dict:
        symbol = state["symbol"]
        print("-" * 60)
        print(f"📊 [Fundamental Analysis Agent] Fetching fundamentals for {symbol}...")

        raw = get_fundamentals(symbol)
        today = raw['date']
        print("📈 [Fundamental Analysis Agent] Preparing data for LLM analysis...")

        # Prompt construction
        prompt = (
            f"You are a fundamental equity analyst for a hedge fund.\n"
            f"Today is {today}. Analyze the following financial information for stock ticker {symbol}:\n\n"
            f"Fundamentals:\n"
            f"- Market Cap: {raw.get('marketCap')}\n"
            f"- Forward P/E: {raw.get('forwardPE')}\n"
            f"- Revenue Growth: {raw.get('revenueGrowth')}\n"
            f"- Profit Margins: {raw.get('profitMargins')}\n"
            f"- Operating Cashflow: {raw.get('operatingCashflow')}\n"
            f"- Free Cashflow: {raw.get('freeCashflow')}\n"
            f"- Debt to Equity: {raw.get('debtToEquity')}\n\n"
            f"Stock Price Trend (past 1 year):\n"
            f"- Start Date: {raw['price_trend']['start_date']}\n"
            f"- End Date: {raw['price_trend']['end_date']}\n"
            f"- Start Price: ${raw['price_trend']['start_price']}\n"
            f"- End Price: ${raw['price_trend']['end_price']}\n"
            f"- Percentage Change: {raw['price_trend']['percent_change']}%\n\n"
            "Please summarize your analysis in a numbered list of 3–5 insights. "
            "Focus on financial strength, valuation, growth potential, and whether the recent price trend supports the fundamentals."
        )

        print("🤖 [Fundamental Analysis Agent] Analyzing fundamentals with LLM...")
        insights = llm.invoke(prompt).content.strip()
        print("✅ [Fundamental Analysis Agent] Fundamental analysis complete.")
        print("-" * 60 + "\n")

        # Format header and final report
        report_header = format_box([
            f"📊 Fundamental Analysis Report for {symbol}",
            f"📅 Date: {today}"
        ], width=90)

        full_report = f"{report_header}\n{insights}\n"
        print(full_report)

        return {
            "symbol": symbol,
            "fundamental_report": full_report
        }

    return RunnableLambda(_invoke)
