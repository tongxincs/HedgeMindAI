from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph
from agents.fundamental_agent import fundamental_agent
from agents.news_agent import news_agent
from agents.sentiment_agent import sentiment_agent
from agents.strategist_agent import strategist_agent
from typing import TypedDict


class GraphState(TypedDict):
    symbol: str
    plot_query: str
    fundamental_report: str
    news_report: str
    sentiment_report: str
    plot_report: str
    strategist_summary: str



def build_graph():
    # Initialize the graph builder
    builder = StateGraph(state_schema=GraphState)

    # Add individual agents as nodes
    builder.add_node("fundamental", fundamental_agent())
    builder.add_node("news", news_agent())
    builder.add_node("sentiment", sentiment_agent())
    builder.add_node("strategist", strategist_agent())

    # Set execution flow
    builder.set_entry_point("fundamental")
    builder.add_edge("fundamental", "news")
    builder.add_edge("news", "sentiment")
    builder.add_edge("sentiment", "strategist")

    # Strategist agent is the final summarizer
    builder.set_finish_point("strategist")

    # Compile the graph into a runnable flow
    return builder.compile()

