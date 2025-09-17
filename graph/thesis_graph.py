from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph
from typing import TypedDict
from agents.fundamental_agent import fundamental_agent
from agents.news_agent import news_agent
from agents.sentiment_agent import sentiment_agent
from agents.strategist_agent import strategist_agent
from agents.satellite_agent import satellite_agent
from agents.earnings_agent import quarterly_earnings_agent
from agents.insider_transaction_agent import insider_transaction_agent
from agents.chart_agent import chart_agent


class GraphState(TypedDict):
    symbol: str
    # Individual agent reports
    fundamental_report: str
    quarterly_report: str
    insider_report: str
    chart_report: str
    news_report: str
    sentiment_report: str
    satellite_report: str
    strategist_summary: str
    # Aggregated text for chart agent
    plot_query: str


def build_graph():
    builder = StateGraph(state_schema=GraphState)

    # ---------------------------
    # Layer 1: Parallel fetchers
    # ---------------------------
    builder.add_node("fundamental", fundamental_agent())
    builder.add_node("quarterly_earnings", quarterly_earnings_agent())
    builder.add_node("insider_transaction", insider_transaction_agent())

    # ---------------------------
    # Layer 2: Aggregator
    # ---------------------------
    def gather_reports(state: dict) -> dict:
        """
        Combine Layer 1 reports into a single plot_query string.
        """
        return {
            **state,
            "plot_query": "\n\n".join(
                [
                    state.get("fundamental_report", ""),
                    state.get("quarterly_report", ""),
                    state.get("insider_report", ""),
                ]
            )
        }

    builder.add_node("gather_reports", gather_reports)

    # ---------------------------
    # Layer 3: Visualization
    # ---------------------------
    builder.add_node("chart_agent", chart_agent())

    # ---------------------------
    # Layer 4: Contextual
    # ---------------------------
    builder.add_node("news", news_agent())
    builder.add_node("sentiment", sentiment_agent())
    builder.add_node("satellite", satellite_agent())

    # ---------------------------
    # Layer 5: Final strategist
    # ---------------------------
    builder.add_node("strategist", strategist_agent())

    # ---------------------------
    # Edges (multi-layer flow)
    # ---------------------------

    # Entry → parallel layer
    builder.set_entry_point("fundamental")
    builder.add_edge("fundamental", "gather_reports")
    builder.add_edge("quarterly_earnings", "gather_reports")
    builder.add_edge("insider_transaction", "gather_reports")

    # Gather → visualization
    builder.add_edge("gather_reports", "chart_agent")

    # Visualization → contextual → strategist
    builder.add_edge("chart_agent", "news")
    builder.add_edge("news", "sentiment")
    builder.add_edge("sentiment", "satellite")
    builder.add_edge("satellite", "strategist")

    # Final step
    builder.set_finish_point("strategist")

    # Compile the graph into a runnable flow
    return builder.compile()

