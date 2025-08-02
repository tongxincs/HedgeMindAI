from dotenv import load_dotenv
load_dotenv()

from agents.sentiment_agent import sentiment_agent


def test_sentiment_agent_returns_summary():
    agent = sentiment_agent()
    state = {"symbol": "TSLA"}

    result = agent.invoke(state)

    assert isinstance(result, dict)
    assert "sentiment_report" in result
    report = result["sentiment_report"]

    assert isinstance(report, str)
    assert len(report) > 0
    assert "Reddit Sentiment Report" in report

    print("\n✅ Sentiment agent test passed.")
    print(report)


if __name__ == "__main__":
    test_sentiment_agent_returns_summary()
