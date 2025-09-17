from typing import Dict, Any
from datetime import datetime
from satellite.router import run_satellite_module
from tools.yahoo_finance import get_fundamentals
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from tools.format import format_box

# Single shared model; temp=0 for deterministic JSON
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _extract_json(text: str) -> str:
    """
    Make sure we return STRICT JSON:
    - Strip markdown fences ```json ... ```
    - Trim whitespace
    - If multiple braces exist, grab the first {...} block
    """
    s = text.strip()
    if s.startswith("```"):
        # Remove leading/trailing fenced block
        s = s.strip("`")
        # after stripping backticks, try to find first { ... } span
    # Try to locate the first JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s  # best effort; caller will json.loads() and raise if invalid


def llm_adapter(*, system: str, user: str) -> str:
    """
    Adapter: accepts system + user strings, returns STRICT JSON string.
    Uses LangChain ChatGoogleGenerativeAI under the hood.
    """
    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    # resp.content is a string; ensure it’s strict JSON for our downstream parser
    return _extract_json(resp.content)

def format_satellite_report(symbol: str, today: str, summary: dict) -> str:
    """
    Convert raw satellite summary dict into a formatted string report.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol.
    today : str
        Date string (YYYY-MM-DD).
    summary : dict
        Satellite summary from run_satellite_module.

    Returns
    -------
    str
        Nicely formatted report string.
    """
    # Header box
    report_header = format_box([
        f"🛰️ Satellite Summary Report for {symbol}",
        f"📅 Date: {today}"
    ], width=90)

    # Extract fields
    headline = summary.get("headline", "No headline")
    bullets = summary.get("bullets", [])
    conf = summary.get("confidence", None)
    attr = summary.get("attribution", [])

    # Format details
    bullet_lines = (
        "\n".join([f"{i+1}. {b}" for i, b in enumerate(bullets)])
        if bullets else "No key points."
    )
    conf_line = f"\nConfidence: {conf:.2f}" if isinstance(conf, (int, float)) else ""
    attr_line = f"\nSources: {', '.join(attr)}" if attr else ""

    return f"{report_header}\n{headline}\n\n{bullet_lines}{conf_line}{attr_line}\n"


def satellite_agent(
    sites_db: Dict[str, list] | None = None,
    proxies_db: Dict[str, list] | None = None,
):
    """
    LangGraph-compatible node factory.
    Runs Plan → Observe → Explain and stores a JSON string in 'satellite_report'.
    """
    sites_db = sites_db or {}
    proxies_db = proxies_db or {}

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        symbol = state.get("symbol")
        industry = get_fundamentals(symbol).get('industry')
        today = datetime.today().strftime("%Y-%m-%d")

        print("-" * 60)
        print(f"🛰️ [Satellite Agent] Running satellite analysis for {symbol} in industry {industry}...")

        summary = run_satellite_module(
            llm_planner=llm_adapter,      # adapter returns raw JSON string
            llm_summarizer=llm_adapter,   # same adapter for summarizer
            ticker=symbol,
            industry=industry,
            sites_db=sites_db,
            proxies_db=proxies_db,
        )

        # Format report
        full_report = format_satellite_report(symbol, today, summary)

        print("✅ [Satellite Agent] Satellite analysis complete.")
        print("-" * 60 + "\n")
        print(full_report)

        return {
            "symbol": symbol,
            "satellite_report": full_report
        }

    return node
