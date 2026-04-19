from typing import TypedDict
from langgraph.graph import StateGraph, END

from src.agents.analyzer_agent import AnalyzerAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.report_agent import ReportAgent


class AgentState(TypedDict):
    player_data: dict
    churn_prob: float
    genre: str
    analysis: str
    strategies: dict
    report: dict
    status: str


def analyze_node(state: AgentState):
    try:
        agent = AnalyzerAgent()
        result = agent.analyze(state['player_data'], state['churn_prob'])
        return {"analysis": result, "status": "analyzing"}
    except Exception as e:
        return {"analysis": "Error in analysis. Proceeding with raw data.", "status": "error_analyzing"}


def strategize_node(state: AgentState):
    try:
        agent = StrategyAgent()
        result = agent.generate(state['analysis'], state['churn_prob'], state['genre'])
        return {"strategies": result, "status": "strategizing"}
    except Exception as e:
        return {
            "strategies": {"strategies": [{"action": "Contact Support", "priority": "Low", "rationale": "System Error"}]},
            "status": "error_strategizing"
        }


def report_node(state: AgentState):
    try:
        agent = ReportAgent()
        result = agent.generate(
            state['player_data'],
            state['churn_prob'],
            state['analysis'],
            state['strategies']
        )
        return {"report": result, "status": "complete"}
    except Exception as e:
        return {
            "report": {"player_profile": {"summary": state['analysis']}, "risk_analysis": {"level": "Unknown"}},
            "status": "error_reporting"
        }


def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("strategize", strategize_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "strategize")
    workflow.add_edge("strategize", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def run_churn_analysis_workflow(player_data: dict, churn_prob: float, genre: str = "RPG"):
    app = build_workflow()
    initial_state = {
        "player_data": player_data,
        "churn_prob": churn_prob,
        "genre": genre,
        "analysis": "",
        "strategies": {},
        "report": {},
        "status": "starting"
    }
    return app.invoke(initial_state)
