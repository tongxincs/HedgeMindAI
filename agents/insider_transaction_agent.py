from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.yahoo_finance import get_insider_transactions_json
from tools.format import format_box

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def insider_transaction_agent() -> RunnableLambda:
    """
    Insider Transaction Analysis Agent:
    Fetches insider transactions for a stock ticker,
    then uses LLM to analyze buying/selling patterns and implications.
    """

    def _invoke(state: dict) -> dict:
        symbol = state["symbol"]
        print("-" * 60)
        print(f"🕵️ [Insider Transaction Agent] Fetching insider transactions for {symbol}...")

        raw = get_insider_transactions_json(symbol, last_n=15)
        today = raw["as_of"]

        print("📈 [Insider Transaction Agent] Preparing data for LLM analysis...")

        # Compact table for context
        tx_lines = []
        for t in raw.get("transactions", []):
            tx_lines.append(
                f"{t.get('date')}: {t.get('filer')} {t.get('transaction')} "
                f"({t.get('shares')} shares @ {t.get('price')})"
            )
        tx_preview = "\n".join(tx_lines[-5:]) if tx_lines else "(No insider transactions found.)"

        # Prompt for LLM
        prompt = (
            f"You are an equity analyst specializing in insider trading.\n"
            f"Today is {today}. Analyze the recent insider transactions for stock ticker {symbol}.\n\n"
            f"Summary:\n"
            f"- Total transactions: {raw['summary']['total_transactions']}\n"
            f"- Net shares: {raw['summary']['net_shares']}\n"
            f"- Total transaction value (USD): {raw['summary']['total_value_usd']}\n"
            f"- Breakdown by type: {raw['summary']['by_type_counts']}\n\n"
            f"Recent transactions:\n{tx_preview}\n\n"
            "Provide a concise 3–5 point analysis of insider activity. "
            "Comment on whether insiders are net buyers or sellers, the scale of the activity, "
            "and what it may suggest about insider confidence in the company."
        )

        print("🤖 [Insider Transaction Agent] Analyzing transactions with LLM...")
        insights = llm.invoke(prompt).content.strip()
        print("✅ [Insider Transaction Agent] Insider transaction analysis complete.")
        print("-" * 60 + "\n")

        report_header = format_box([
            f"🕵️ Insider Transaction Report for {symbol}",
            f"📅 Date: {today}"
        ], width=90)

        full_report = f"{report_header}\n{insights}\n"
        print(full_report)

        return {
            "symbol": symbol,
            "insider_report": full_report
        }

    return RunnableLambda(_invoke)
