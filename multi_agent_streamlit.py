import streamlit as st
import logging
from typing import TypedDict, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import logging
logging.basicConfig(level=logging.INFO)

logging.info("Planner Agent started")


# =========================
# Logging Configuration
# =========================
logging.basicConfig(level=logging.INFO)

# =========================
# Application Overview
# =========================
"""
Multi-Agent Research System – Architecture Overview

Agents and Responsibilities:
- Researcher Agent:
    Gathers external information using search tools (Tavily).
- Analyzer Agent:
    Processes research findings to extract insights and patterns.
- Writer Agent:
    Synthesizes research and analysis into a structured final report.
- Supervisor Agent:
    Controls execution order and routes tasks between agents
    using deterministic logic (no LLM-based routing).

Execution Flow:
Researcher → Analyzer → Writer → FINISH
"""

# =========================
# State Definition
# =========================
class ResearchState(TypedDict):
    messages: list
    research_topic: str
    research_findings: str
    analysis: str
    final_report: str
    next_agent: Optional[str]

# =========================
# LLM & Tools
# =========================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = TavilySearchResults(max_results=5)

# =========================
# Agent Prompt Definitions
# =========================
RESEARCHER_PROMPT = """
You are a research agent.
Search for factual, reliable information about the given topic.
Summarize key findings concisely.
"""

ANALYZER_PROMPT = """
You are an analysis agent.
Analyze the research findings.
Identify trends, insights, and important observations.
"""

WRITER_PROMPT = """
You are a writing agent.
Create a clear, well-structured final report
based on the research and analysis.
"""

# =========================
# Agent Node Implementations
# =========================
def researcher_node(state: ResearchState, status_placeholder):
    """
    Researcher Agent

    Responsibility:
    - Searches for relevant information about the research topic
    - Uses external tools (TavilySearch)
    - Produces summarized research findings

    Input:
    - research_topic from state

    Output:
    - research_findings added to state
    """
    status_placeholder.info("🔍 Researcher Agent: Gathering information...")
    results = search_tool.invoke(state["research_topic"])

    summary = "\n".join([r["content"] for r in results])
    return {
        **state,
        "research_findings": summary,
        "next_agent": "analyzer"
    }


def analyzer_node(state: ResearchState, status_placeholder):
    """
    Analyzer Agent

    Responsibility:
    - Reviews research findings
    - Identifies trends, insights, and contradictions
    - Produces structured analytical output

    Input:
    - research_findings from state

    Output:
    - analysis added to state
    """
    status_placeholder.info("📊 Analyzer Agent: Analyzing findings...")
    response = llm.invoke(
        ANALYZER_PROMPT + "\n\n" + state["research_findings"]
    )

    return {
        **state,
        "analysis": response.content,
        "next_agent": "writer"
    }


def writer_node(state: ResearchState, status_placeholder):
    """
    Writer Agent

    Responsibility:
    - Combines research findings and analysis
    - Generates a polished, human-readable report
    - Produces final output for the user

    Input:
    - research_findings
    - analysis

    Output:
    - final_report added to state
    """
    status_placeholder.info("✍️ Writer Agent: Writing final report...")
    response = llm.invoke(
        WRITER_PROMPT
        + "\n\nResearch:\n"
        + state["research_findings"]
        + "\n\nAnalysis:\n"
        + state["analysis"]
    )

    return {
        **state,
        "final_report": response.content,
        "next_agent": None
    }


def supervisor_node(state: ResearchState):
    """
    Supervisor Agent

    Responsibility:
    - Determines which agent should run next
    - Enforces execution order:
      Researcher → Analyzer → Writer → FINISH
    - Uses deterministic logic instead of an LLM
      for predictable agent routing
    """
    if state["next_agent"] == "researcher":
        return "researcher"
    if state["next_agent"] == "analyzer":
        return "analyzer"
    if state["next_agent"] == "writer":
        return "writer"
    return END

# =========================
# Workflow Graph Construction
# =========================
def create_research_workflow(status_placeholder):
    graph = StateGraph(ResearchState)

    graph.add_node("researcher", lambda s: researcher_node(s, status_placeholder))
    graph.add_node("analyzer", lambda s: analyzer_node(s, status_placeholder))
    graph.add_node("writer", lambda s: writer_node(s, status_placeholder))

    graph.add_conditional_edges(
        "researcher",
        supervisor_node,
        {"analyzer": "analyzer"}
    )
    graph.add_conditional_edges(
        "analyzer",
        supervisor_node,
        {"writer": "writer"}
    )
    graph.add_conditional_edges(
        "writer",
        supervisor_node,
        {END: END}
    )

    graph.set_entry_point("researcher")
    return graph.compile()

# =========================
# Streamlit Application UI
# =========================
def main():
    st.title("🧠 Multi-Agent Research System")

    research_topic = st.text_input("Enter a research topic:")

    if st.button("Run Research") and research_topic:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        try:
            with st.spinner("Initializing research workflow..."):
                progress_bar.progress(10)

                app = create_research_workflow(status_placeholder)

                initial_state: ResearchState = {
                    "messages": [HumanMessage(content=f"Research topic: {research_topic}")],
                    "research_topic": research_topic,
                    "research_findings": "",
                    "analysis": "",
                    "final_report": "",
                    "next_agent": "researcher"
                }

                result = app.invoke(initial_state)

                progress_bar.progress(100)

            st.success("✅ Research completed successfully!")

            st.subheader("📄 Final Report")
            st.write(result["final_report"])

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")


if __name__ == "__main__":
    main()
