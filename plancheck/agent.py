"""The LangGraph workflow.

    parse_rules ─► extract ─► snap ─► validate ──(geometry errors, attempts left)──► extract (repair)
                                          │
                                          └──(clean or out of attempts)──► check ─► END

- parse_rules: written spec -> typed rules (LLM structured output), or a YAML file offline.
- extract:     vision-language model -> FloorPlan (structured output); on repair it gets the errors back.
- snap:        moves coordinates onto the wall lines and door gaps detected in the full-resolution image.
- validate:    exact geometry sanity checks (invalid polygons, overlaps, gaps, doors off walls).
- check:       every rule decided by Shapely/NetworkX, never by the model.
"""

from __future__ import annotations

import operator
from dataclasses import asdict
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from PIL import Image

from .extract import Extractor, Usage
from .geometry import PlanGeometry
from .parse_rules import parse_rules_text
from .rules import RuleResult, RuleSet, check_all
from .schema import FloorPlan, PlanMeta
from .snap import snap_plan, uncovered_regions


class State(TypedDict, total=False):
    image_path: str
    meta: PlanMeta
    rules_text: str | None
    rules: RuleSet | None
    plan: FloorPlan | None
    issues: list[dict]
    attempts: int
    results: list[RuleResult]
    geometry: dict
    usage: Annotated[list[dict], operator.add]
    log: Annotated[list[str], operator.add]


def build_graph(extractor: Extractor, rules_model: str | None = None, max_attempts: int = 3,
                use_snap: bool = True):
    def parse_rules(state: State) -> dict:
        if state.get("rules") is not None:
            return {"log": [f"rules: {len(state['rules'].rules)} loaded from file"]}
        if not rules_model:
            raise ValueError("rules_text given but no rules_model to parse it")
        rs, u = parse_rules_text(state["rules_text"], rules_model)
        return {"rules": rs, "usage": [{"step": "parse_rules", "model": rules_model, **asdict(u)}],
                "log": [f"rules: parsed {len(rs.rules)} rules with {rules_model}"]}

    def extract(state: State) -> dict:
        n = state.get("attempts", 0) + 1
        errors = [i for i in state.get("issues", []) if i["severity"] == "error"]
        feedback = "\n".join(f"- {i['message']}" for i in errors) if errors and n > 1 else None
        plan, u = extractor.extract(state["image_path"], state["meta"], feedback=feedback,
                                    previous=state.get("plan") if feedback else None)
        what = "repair" if feedback else "extract"
        return {"plan": plan, "attempts": n,
                "usage": [{"step": what, "model": extractor.name, **asdict(u)}],
                "log": [f"{what} #{n}: {len(plan.rooms)} rooms, {len(plan.doors)} doors ({extractor.name})"]}

    def snap(state: State) -> dict:
        if not use_snap:
            return {}
        img = Image.open(state["image_path"])
        plan, stats = snap_plan(state["plan"], img, state["meta"].px_per_m)
        return {"plan": plan, "log": [f"snap: {stats['wall_lines']} wall lines, "
                                      f"mean vertex shift {stats['mean_vertex_shift_px']} px"]}

    def validate(state: State) -> dict:
        geo = PlanGeometry(state["plan"], state["meta"])
        issues = [asdict(i) for i in geo.validate()]
        if use_snap:  # image-based check: drawn floor area that no room covers
            for u in uncovered_regions(state["plan"], Image.open(state["image_path"]), state["meta"].px_per_m):
                issues.append({"severity": "error", "code": "uncovered",
                               "message": f"A drawn area of about {u['area_m2']} m² around ({u['x']:.0f}, {u['y']:.0f}) px "
                                          "is not covered by any room: a room is probably missing.",
                               "subject": None})
        n_err = sum(i["severity"] == "error" for i in issues)
        return {"issues": issues, "log": [f"validate: {n_err} errors, {len(issues) - n_err} warnings"]}

    def route(state: State) -> str:
        has_err = any(i["severity"] == "error" for i in state["issues"])
        return "extract" if has_err and state["attempts"] < max_attempts else "check"

    def check(state: State) -> dict:
        geo = PlanGeometry(state["plan"], state["meta"])
        results = check_all(state["rules"], geo)
        n_fail = sum(not r.passed for r in results)
        return {"results": results, "geometry": geo.summary(),
                "log": [f"check: {len(results) - n_fail} passed, {n_fail} failed"]}

    g = StateGraph(State)
    for name, fn in [("parse_rules", parse_rules), ("extract", extract), ("snap", snap),
                     ("validate", validate), ("check", check)]:
        g.add_node(name, fn)
    g.add_edge(START, "parse_rules")
    g.add_edge("parse_rules", "extract")
    g.add_edge("extract", "snap")
    g.add_edge("snap", "validate")
    g.add_conditional_edges("validate", route, {"extract": "extract", "check": "check"})
    g.add_edge("check", END)
    return g.compile()


def run_plan(image_path: str, meta: PlanMeta, extractor: Extractor, rules: RuleSet | None = None,
             rules_text: str | None = None, rules_model: str | None = None, max_attempts: int = 3,
             use_snap: bool = True) -> dict[str, Any]:
    graph = build_graph(extractor, rules_model=rules_model, max_attempts=max_attempts, use_snap=use_snap)
    return graph.invoke({"image_path": str(image_path), "meta": meta, "rules": rules, "rules_text": rules_text,
                         "attempts": 0, "issues": [], "usage": [], "log": []})


def total_usage(state: dict) -> Usage:
    u = Usage()
    for x in state.get("usage", []):
        u = u.add(Usage(x["input_tokens"], x["output_tokens"], x["seconds"], x["calls"]))
    return u
