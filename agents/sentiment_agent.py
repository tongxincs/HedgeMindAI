from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.format import format_box
from tools.reddit_scraper import search_posts_by_ticker
from datetime import datetime

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
subreddits = ["wallstreetbets", "stocks"]


def sentiment_agent() -> RunnableLambda:
    """
    Reddit Sentiment Agent:
    Fetches recent Reddit posts about a given stock ticker and summarizes the
    overall sentiment using a language model.

    Returns:
        RunnableLambda that accepts a state dict with key "symbol" and returns:
        {
            "symbol": ...,
            "sentiment_report": formatted summary string
        }
    """
    def _invoke(state: dict) -> dict:
        symbol = state["symbol"]
        today = datetime.now().strftime("%Y-%m-%d")

        print("-" * 60)
        print(f"🔎 [Sentiment Analysis Agent] Fetching Reddit posts for {symbol}...")
        posts = search_posts_by_ticker(
            subreddits=subreddits,
            ticker=symbol,
            days_back=90,
            limit=300
        )

        if not posts:
            print("⚠️ No Reddit posts found.")
            return {
                "symbol": symbol,
                "sentiment_report": f"No recent Reddit sentiment found for {symbol}."
            }

        print(f"📄 [Sentiment Analysis Agent] Retrieved {len(posts)} Reddit posts.")

        # Combine posts into a single LLM-friendly block
        combined = "\n\n".join(
            f"Title: {p['title']}\nBody: {p['body']}" for p in posts[:100]
        )

        # Construct analysis prompt
        prompt = (
            f"You are a social sentiment analyst at a hedge fund.\n"
            f"Analyze the following Reddit posts from r/wallstreetbets and r/stocks about stock ticker {symbol}. "
            f"The posts span the past 3 days.\n\n"
            f"{combined}\n\n"
            "Your task is to analyze Reddit sentiment and user behavior and produce a professional report.\n"
            "Please answer the following:\n\n"
            "1. **Mention Frequency**: Are there many posts mentioning this ticker in the last 7 days? Does it appear to be a trending or high-interest topic?\n\n"
            "2. **Language Style**: Do users use emotionally charged or exaggerated phrases like 'ALL IN', 'YOLO', 'to the moon', or similar? What does that imply about their sentiment or confidence?\n\n"
            "3. **Actual Positions**: Do multiple users indicate they have placed trades recently (within the last 7 days)? What types of trades (calls, puts, shares, etc.) are mentioned?\n\n"
            "4. **Summary**: Based on the above, summarize the overall sentiment (bullish, bearish, or mixed) and behavioral tone of the community in 3–5 bullet points. Be concise and investor-focused."
        )

        print("🤖 [Sentiment Analysis Agent] LLM analyzing sentiment...")
        summary = llm.invoke(prompt).content.strip()
        print("✅ [Sentiment Analysis Agent] Sentiment analysis complete.")
        print("-" * 60 + "\n")

        report_header = format_box([
            f"🧠 Reddit Sentiment Report for {symbol}",
            f"🗓️ Date: {today}"
        ], width=90)

        full_report = f"{report_header}\n{summary}\n"
        print(full_report)

        return {
            "symbol": symbol,
            "sentiment_report": full_report
        }

    return RunnableLambda(_invoke)
