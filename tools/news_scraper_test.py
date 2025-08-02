from tools.news_scraper import fetch_news_articles
from datetime import datetime, timedelta


def test_returns_articles_for_valid_symbol():
    """Test that the scraper returns at least one article for a valid stock ticker."""
    articles = fetch_news_articles("TSLA", max_articles=3, days_ago=30)
    print(f"Fetched {len(articles)} articles.")
    assert len(articles) > 0
    for a in articles:
        assert "title" in a
        assert "summary" in a
        assert "url" in a
        assert "datetime" in a
        assert len(a["summary"]) > 50


def test_respects_max_articles():
    """Test that the scraper does not exceed the requested max_articles limit."""
    articles = fetch_news_articles("AAPL", max_articles=2, days_ago=7)
    assert len(articles) <= 2


def test_filters_by_days_ago():
    """Test that only articles within the time range are returned."""
    articles = fetch_news_articles("MSFT", max_articles=5, days_ago=1)
    cutoff = datetime.now() - timedelta(days=1)
    for a in articles:
        published = datetime.fromisoformat(a["datetime"])
        assert published >= cutoff


def test_returns_empty_for_invalid_symbol():
    """Test that an invalid ticker returns an empty result."""
    articles = fetch_news_articles("XYZINVALIDTICKER123", max_articles=3, days_ago=7)
    assert isinstance(articles, list)
    assert len(articles) == 0


if __name__ == "__main__":
    print("Running scraper tests...\n")

    test_returns_articles_for_valid_symbol()
    print("✅ test_returns_articles_for_valid_symbol passed")

    test_respects_max_articles()
    print("✅ test_respects_max_articles passed")

    test_filters_by_days_ago()
    print("✅ test_filters_by_days_ago passed")

    test_returns_empty_for_invalid_symbol()
    print("✅ test_returns_empty_for_invalid_symbol passed")

    print("\n✅ All tests passed.")
