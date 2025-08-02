from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.news_scraper import fetch_news_articles
from tools.format import format_box
from datetime import datetime

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def news_agent() -> RunnableLambda:
    """
    News Agent:
    Retrieves recent news articles related to a stock ticker and uses LLM
    to summarize sentiment and key investor-relevant themes.

    Returns:
        RunnableLambda accepting a state dict with "symbol", returns:
        {
            "symbol": ...,
            "news_report": formatted summary string
        }
    """

    def _invoke(state: dict) -> dict:
        symbol = state["symbol"]
        today = datetime.today().strftime("%Y-%m-%d")

        print("-" * 60)
        print(f"📰 [News Analysis Agent] Fetching news articles for {symbol}...")
        articles = fetch_news_articles(symbol, max_articles=5, days_ago=7)

        if not articles:
            print("⚠️ No news articles found.")
            return {
                "symbol": symbol,
                "news_report": f"No recent news articles found for {symbol}."
            }

        print(f"📄 [News Analysis Agent] Retrieved {len(articles)} articles. Sending to LLM for analysis...")

        # Prepare input for LLM
        combined_text = "\n\n".join(
            f"Title: {a['title']}\nDate: {a['published']}\n{a['summary']}"
            for a in articles
        )

        prompt = (
            f"You are a financial news analyst.\n"
            f"Analyze the following news articles about stock ticker {symbol}:\n\n"
            f"{combined_text}\n\n"
            "Please summarize the overall sentiment and extract 3 to 5 key investor-relevant insights.\n"
            "Respond in a numbered list format, like:\n"
            "1. ...\n2. ...\n3. ..."
        )

        print("🤖 [News Analysis Agent] Analyzing news content with LLM...")
        insights = llm.invoke(prompt).content.strip()
        print("✅ [News Analysis Agent] News analysis complete.")
        print("-" * 60 + "\n")

        report_header = format_box([
            f"📰 News Summary Report for {symbol}",
            f"📅 Date: {today}"
        ], width=90)

        full_report = f"{report_header}\n{insights}\n"
        print(full_report)

        return {
            "symbol": symbol,
            "news_report": full_report
        }

    return RunnableLambda(_invoke)
