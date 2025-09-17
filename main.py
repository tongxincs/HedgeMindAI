from graph.thesis_graph import build_graph
from tools.format import format_box
from dotenv import load_dotenv

load_dotenv()


def welcome():
    content = [
        "📈 Welcome to HedgeMind AI",
        "",
        "A multi-agent, multi-modal research system for U.S. equities.",
        "It integrates fundamentals, earnings, insider activity, news, sentiment,",
        "and strategy to deliver professional-grade investment insights."
    ]
    print("\n" + format_box(content, width=90) + "\n")


def main():
    welcome()

    symbol = input("🔍 Enter a stock ticker (e.g., TSLA, GOOG): ").upper().strip()
    if not symbol:
        print("❌ No symbol entered. Exiting.")
        return

    message = f"📡 Running multi-agent analysis for {symbol.upper()}..."
    print("\n" + format_box([message], width=90) + "\n")
    graph = build_graph()
    graph.invoke({
        "symbol": symbol,
    })


if __name__ == "__main__":
    main()
