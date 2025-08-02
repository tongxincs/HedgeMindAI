import praw
from datetime import datetime, timedelta, UTC
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="StockInsightAI/1.0"
)


def search_posts_by_ticker(
    subreddits: list,
    ticker: str,
    days_back: int = 7,
    limit: int = 500
) -> list:
    """
    Search recent Reddit posts that mention a stock ticker.

    Args:
        subreddits (list): List of subreddit names (e.g., ['wallstreetbets']).
        ticker (str): Stock ticker to search for.
        days_back (int): Time window to filter posts.
        limit (int): Number of posts to retrieve per subreddit.

    Returns:
        List[dict]: Post data with subreddit, title, body, author, and created date.
    """
    end_time = datetime.now(UTC).date()
    start_time = end_time - timedelta(days=days_back)

    matched_posts = []

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        query = ticker
        for submission in subreddit.search(query, sort="new", limit=limit):
            post_time = datetime.fromtimestamp(submission.created_utc, UTC).date()
            if start_time <= post_time <= end_time:
                if submission.selftext.lower() in ["[removed]", "[deleted]"]:
                    continue  # skip removed content
                matched_posts.append({
                    "subreddit": subreddit_name,
                    "title": submission.title,
                    "body": submission.selftext,
                    "author": str(submission.author),
                    "created": post_time.isoformat()
                })

    return matched_posts
