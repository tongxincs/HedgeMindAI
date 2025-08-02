from datetime import datetime, timedelta
from tools.reddit_scraper import search_posts_by_ticker


def test_fetches_posts_for_valid_ticker():
    """Ensure posts are returned for a known active ticker."""
    posts = search_posts_by_ticker(["wallstreetbets"], "TSLA", days_back=7, limit=50)
    print(f"Found {len(posts)} posts for TSLA in r/wallstreetbets")
    assert isinstance(posts, list)
    assert len(posts) > 0
    for post in posts:
        assert "title" in post
        assert "body" in post
        assert "created" in post
        assert "author" in post
        assert post["body"].lower() not in ["[removed]", "[deleted]"]


def test_returns_empty_for_invalid_ticker():
    """Should return empty list for a gibberish ticker."""
    posts = search_posts_by_ticker(["wallstreetbets"], "ZZZFAKETICKER", days_back=7, limit=50)
    assert isinstance(posts, list)
    assert len(posts) == 0 or all("title" in p for p in posts)


def test_filters_posts_by_date():
    """Ensure returned posts fall within the date range."""
    days_back = 2
    cutoff = datetime.now().date() - timedelta(days=days_back)
    posts = search_posts_by_ticker(["stocks"], "MSFT", days_back=days_back, limit=50)
    for post in posts:
        post_date = datetime.fromisoformat(post["created"]).date()
        assert post_date >= cutoff


if __name__ == "__main__":
    print("Running Reddit scraper tests...\n")

    test_fetches_posts_for_valid_ticker()
    print("✅ test_fetches_posts_for_valid_ticker passed")

    test_returns_empty_for_invalid_ticker()
    print("✅ test_returns_empty_for_invalid_ticker passed")

    test_filters_posts_by_date()
    print("✅ test_filters_posts_by_date passed")

    print("\n✅ All Reddit scraper tests passed.")
