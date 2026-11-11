"""Measure the whole pipeline against labelled plans and compare models on accuracy and cost.

Metrics per model:
- rule_acc:   share of rule verdicts (pass/fail) that match the verdict computed on the ground truth
- plan_exact: share of plans where every rule verdict is right
- room_f1:    rooms matched to ground truth at IoU >= 0.5 (precision/recall/F1), mean IoU of matches
- type_acc:   room type correct among matched rooms
- area_err:   mean absolute area error (%) among matched rooms
- door_recall: ground-truth doors with an extracted door within 0.3 m
- attempts, seconds, tokens and cost per plan (cost needs prices in prices.yaml)
"""

from __future__ import annotations

import json
import math
import statistics as st
import traceback
from pathlib import Path

import yaml
from shapely.geometry import Point, Polygon

from .agent import run_plan, total_usage
from .extract import make_extractor
from .geometry import PlanGeometry
from .rules import RuleSet, check_all
from .schema import FloorPlan, PlanMeta
from .synth import load_truth


def match_rooms(pred: FloorPlan, truth: FloorPlan, thr: float = 0.5) -> list[tuple[str, str, float]]:
    P = {r.id: Polygon(r.polygon).buffer(0) for r in pred.rooms}
    T = {r.id: Polygon(r.polygon).buffer(0) for r in truth.rooms}
    pairs = []
    for t, tp in T.items():
        for p, pp in P.items():
            inter = tp.intersection(pp).area
            if inter:
                pairs.append((inter / tp.union(pp).area, t, p))
    used_t, used_p, out = set(), set(), []
    for iou, t, p in sorted(pairs, reverse=True):
        if iou >= thr and t not in used_t and p not in used_p:
            used_t.add(t), used_p.add(p), out.append((t, p, iou))
    return out


def score_plan(state: dict, truth: FloorPlan, meta: PlanMeta, rules: RuleSet) -> dict:
    true_res = {r.rule_id: r.passed for r in check_all(rules, PlanGeometry(truth, meta))}
    got = {r.rule_id: r.passed for r in state["results"]}
    right = sum(true_res[k] == got.get(k) for k in true_res)
    pred = state["plan"]
    m = match_rooms(pred, truth)
    tp = len(m)
    prec = tp / len(pred.rooms) if pred.rooms else 0.0
    rec = tp / len(truth.rooms) if truth.rooms else 0.0
    ttype = {r.id: r.type for r in truth.rooms}
    ptype = {r.id: r.type for r in pred.rooms}
    tpoly = {r.id: Polygon(r.polygon) for r in truth.rooms}
    ppoly = {r.id: Polygon(r.polygon) for r in pred.rooms}
    tol = 0.3 * meta.px_per_m
    pdoors = [Point(d.position) for d in pred.doors]
    door_hit = sum(any(Point(d.position).distance(q) <= tol for q in pdoors) for d in truth.doors)
    u = total_usage(state)
    return {
        "rule_correct": right, "rule_total": len(true_res),
        "wrong_rules": sorted(k for k in true_res if true_res[k] != got.get(k)),
        "room_precision": prec, "room_recall": rec,
        "room_f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
        "mean_iou": st.mean([i for *_, i in m]) if m else 0.0,
        "type_acc": st.mean([ttype[t] == ptype[p] for t, p, _ in m]) if m else 0.0,
        "area_err_pct": st.mean([abs(ppoly[p].area - tpoly[t].area) / tpoly[t].area * 100 for t, p, _ in m]) if m else 100.0,
        "door_recall": door_hit / len(truth.doors) if truth.doors else 1.0,
        "attempts": state["attempts"], "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
        "seconds": u.seconds,
    }


def _cost(row: dict, price: dict | None) -> float | None:
    if not price:
        return None
    return (row["input_tokens"] * price["input_per_mtok"] + row["output_tokens"] * price["output_per_mtok"]) / 1e6


def evaluate(plans_dir: str | Path, models: list[str], rules: RuleSet, out_dir: str | Path,
             limit: int | None = None, prices_path: str | Path | None = None, use_snap: bool = True,
             max_attempts: int = 3) -> list[dict]:
    plans = sorted(Path(plans_dir).glob("*.png"))[:limit]
    prices = yaml.safe_load(Path(prices_path).read_text()) if prices_path and Path(prices_path).exists() else {}
    prices = prices or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for spec in models:
        ex = make_extractor(spec)
        rows = []
        with (out / f"{spec.replace(':', '_').replace('/', '_')}.jsonl").open("w") as fh:
            for p in plans:
                truth, meta = load_truth(p)
                try:
                    state = run_plan(str(p), meta, ex, rules=rules, max_attempts=max_attempts, use_snap=use_snap)
                    row = {"plan": p.stem, "ok": True, **score_plan(state, truth, meta, rules)}
                except Exception as e:  # a crash counts as every verdict wrong
                    row = {"plan": p.stem, "ok": False, "error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc(limit=3), "rule_correct": 0, "rule_total": len(rules.rules)}
                row["cost_usd"] = _cost(row, prices.get(spec)) if row["ok"] else None
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                print(f"[{spec}] {p.stem}: rules {row['rule_correct']}/{row['rule_total']}"
                      + ("" if row["ok"] else f"  ERROR {row['error']}"), flush=True)
        ok = [r for r in rows if r["ok"]]
        mean = lambda k: st.mean([r[k] for r in ok]) if ok else math.nan  # noqa: E731
        costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]
        summary.append({
            "model": spec, "plans": len(rows), "errors": len(rows) - len(ok),
            "rule_acc": sum(r["rule_correct"] for r in rows) / max(1, sum(r["rule_total"] for r in rows)),
            "plan_exact": sum(r["rule_correct"] == r["rule_total"] for r in rows) / max(1, len(rows)),
            "room_f1": mean("room_f1"), "mean_iou": mean("mean_iou"), "type_acc": mean("type_acc"),
            "area_err_pct": mean("area_err_pct"), "door_recall": mean("door_recall"),
            "attempts": mean("attempts"), "seconds": mean("seconds"),
            "tokens": mean("input_tokens") + mean("output_tokens") if ok else math.nan,
            "cost_usd": st.mean(costs) if costs else None,
        })
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(summary_table(summary))
    return summary


def summary_table(summary: list[dict]) -> str:
    head = ("| Model | Plans | Rule acc. | Plans fully right | Room F1 | Mean IoU | Type acc. | Area err. | "
            "Door recall | Attempts | Sec/plan | Tokens/plan | $/plan |")
    lines = [head, "|" + "---|" * 13]
    for s in summary:
        cost = "—" if s["cost_usd"] is None else f"{s['cost_usd']:.4f}"
        lines.append(
            f"| {s['model']} | {s['plans']} | {s['rule_acc']:.1%} | {s['plan_exact']:.1%} | {s['room_f1']:.3f} | "
            f"{s['mean_iou']:.3f} | {s['type_acc']:.1%} | {s['area_err_pct']:.1f}% | {s['door_recall']:.1%} | "
            f"{s['attempts']:.2f} | {s['seconds']:.1f} | {s['tokens']:.0f} | {cost} |")
    return "\n".join(lines) + "\n"
