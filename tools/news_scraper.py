from datetime import datetime, timedelta, UTC
from tools.finnhub_client import finnhub_client


def fetch_news_articles(symbol: str, max_articles: int = 5, days_ago: int = 7):
    """
    Fetches recent company news using Finnhub API for a given stock ticker.

    Args:
        symbol (str): Stock ticker symbol (e.g. "AAPL").
        max_articles (int): Maximum number of articles to return.
        days_ago (int): Number of days back to look for news.

    Returns:
        List[dict]: List of news articles with title, summary, source, and URL.
    """
    to_date = datetime.now(UTC).date()
    from_date = to_date - timedelta(days=days_ago)

    response = finnhub_client.company_news(
        symbol,
        _from=from_date.strftime("%Y-%m-%d"),
        to=to_date.strftime("%Y-%m-%d"),
    )

    articles = []
    for item in response:
        if not item.get("summary"):
            continue
        articles.append({
            "title": item["headline"],
            "summary": item["summary"],
            "source": item.get("source", ""),
            "url": item["url"],
            "published": datetime.fromtimestamp(item["datetime"], UTC).strftime("%Y-%m-%d %H:%M:%S"),
        })
        if len(articles) >= max_articles:
            break

    return articles
