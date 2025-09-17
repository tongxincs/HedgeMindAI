from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.yahoo_finance import get_quarterly_earnings_json
from tools.format import format_box

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def quarterly_earnings_agent() -> RunnableLambda:
    """
    Quarterly Earnings Analysis Agent:
    Fetches last 4 quarters of earnings and growth,
    then uses LLM to generate insights on trends and valuation.
    """

    def _invoke(state: dict) -> dict:
        symbol = state["symbol"]
        print("-" * 60)
        print(f"📑 [Quarterly Earnings Agent] Fetching quarterly earnings for {symbol}...")

        raw = get_quarterly_earnings_json(symbol, max_quarters=8)
        today = raw["as_of"]

        print("📈 [Quarterly Earnings Agent] Preparing data for LLM analysis...")

        # Prepare a compact table for LLM
        q_lines = []
        for q in raw.get("quarters", []):
            q_lines.append(
                f"{q['period']}: Rev={q.get('revenue')} "
                f"Net={q.get('net_income')} EPS={q.get('eps_diluted')} "
                f"(Rev QoQ={q['growth']['revenue'].get('qoq')}, YoY={q['growth']['revenue'].get('yoy')})"
            )
        q_preview = "\n".join(q_lines) if q_lines else "(No quarterly data found.)"

        prompt = (
            f"You are an equity analyst preparing an earnings trend analysis.\n"
            f"Today is {today}. Review the last 4 quarters of results for stock ticker {symbol}:\n\n"
            f"{q_preview}\n\n"
            "Provide a 3–5 point analysis of the company’s recent performance. "
            "Focus on revenue growth, profitability, EPS trends, and whether earnings momentum is strengthening or weakening."
        )

        print("🤖 [Quarterly Earnings Agent] Analyzing earnings with LLM...")
        insights = llm.invoke(prompt).content.strip()
        print("✅ [Quarterly Earnings Agent] Quarterly earnings analysis complete.")
        print("-" * 60 + "\n")

        report_header = format_box([
            f"📑 Quarterly Earnings Report for {symbol}",
            f"📅 Date: {today}"
        ], width=90)

        full_report = f"{report_header}\n{insights}\n"
        print(full_report)

        return {
            "symbol": symbol,
            "earnings_report": full_report
        }

    return RunnableLambda(_invoke)
