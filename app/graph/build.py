import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from app.graph import judge, policy
from app.rag import retrieval
from app.core.obs import log_event, traced_node
from app.models.schemas import Chunk, CoverageItem, Decision


class ScreeningState(TypedDict):
    thread_id: str
    language: str
    turns: Annotated[list[str], operator.add]
    claims_summary_en: str
    retrieved: list[dict]
    coverage: list[dict]
    decision: dict | None
    followups_used: int
    score: int | None
    justification: str | None
    followup_question: str | None


@traced_node
def summarize_node(state: ScreeningState) -> dict:
    summary = judge.summarize(state["turns"], state.get("followup_question"), state["thread_id"])
    return {"claims_summary_en": summary}


@traced_node
def retrieve_node(state: ScreeningState) -> dict:
    retrieved = retrieval.retrieve(state["claims_summary_en"], state["thread_id"])
    return {"retrieved": [chunk.model_dump(mode="json") for chunk in retrieved]}


@traced_node
def analyze_node(state: ScreeningState) -> dict:
    retrieved = [Chunk.model_validate(chunk) for chunk in state["retrieved"]]
    coverage = judge.analyze_coverage(state["turns"], retrieved, state["language"], state["thread_id"])
    decision = policy.decide(coverage, state["followups_used"])
    log_event("decision.made", state["thread_id"], probe=decision.probe, target=decision.target,
              reason=decision.reason, followups_used=state["followups_used"])
    return {
        "coverage": [item.model_dump(mode="json") for item in coverage],
        "decision": decision.model_dump(mode="json"),
    }


def route_after_analysis(state: ScreeningState) -> str:
    return "probe" if Decision.model_validate(state["decision"]).probe else "score"


@traced_node
def compose_probe_node(state: ScreeningState) -> dict:
    decision = Decision.model_validate(state["decision"])
    retrieved = [Chunk.model_validate(chunk) for chunk in state["retrieved"]]
    question = judge.write_followup(decision.target, retrieved,
                                     state["language"], state["thread_id"])
    log_event("interrupt.raised", state["thread_id"], target=decision.target,
              followup_question=question)
    return {"followup_question": question}


@traced_node
def probe_node(state: ScreeningState) -> dict:
    # No code before interrupt(): LangGraph replays this node from the top on resume, so
    # anything here (e.g. the write_followup() LLM call) would re-fire on every resume.
    decision = Decision.model_validate(state["decision"])
    answer = interrupt({"question": state["followup_question"], "target": decision.target})
    return {"turns": [answer], "followups_used": state["followups_used"] + 1}


@traced_node
def score_node(state: ScreeningState) -> dict:
    coverage = [CoverageItem.model_validate(item) for item in state["coverage"]]
    retrieved = [Chunk.model_validate(chunk) for chunk in state["retrieved"]]
    score, justification = judge.score(state["turns"], coverage, retrieved,
                                        state["language"], state["thread_id"])
    return {"score": score, "justification": justification}


_g = StateGraph(ScreeningState)
for _name, _fn in [("summarize", summarize_node), ("retrieve", retrieve_node),
                    ("analyze", analyze_node), ("compose_probe", compose_probe_node),
                    ("probe", probe_node), ("score", score_node)]:
    _g.add_node(_name, _fn)
_g.add_edge(START, "summarize")
_g.add_edge("summarize", "retrieve")
_g.add_edge("retrieve", "analyze")
_g.add_conditional_edges("analyze", route_after_analysis, {"probe": "compose_probe", "score": "score"})
_g.add_edge("compose_probe", "probe")
_g.add_edge("probe", "summarize")
_g.add_edge("score", END)

graph = _g.compile(checkpointer=InMemorySaver())
