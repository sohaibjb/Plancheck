"""Command line: plancheck synth | check | eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import run_plan
from .evaluate import evaluate
from .extract import make_extractor
from .parse_rules import load_rules_yaml
from .report import markdown, overlay
from .schema import PlanMeta
from .synth import make_dataset


def _meta(image: str, px_per_m: float | None) -> PlanMeta:
    if px_per_m:
        return PlanMeta(px_per_m=px_per_m)
    side = Path(image).with_suffix(".json")
    if side.exists():
        return PlanMeta.model_validate(json.loads(side.read_text())["meta"])
    raise SystemExit("--px-per-m is required (pixels per metre on the drawing)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="plancheck", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate labelled synthetic floor plans")
    s.add_argument("--out", default="data/plans")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--px-per-m", type=float, default=120.0)

    c = sub.add_parser("check", help="check one plan image against a specification")
    c.add_argument("image")
    c.add_argument("--rules", default="data/rules/residential.yaml", help=".yaml (typed) or .txt (parsed by --rules-model)")
    c.add_argument("--model", default="simulated", help="e.g. anthropic:claude-sonnet-5-5, openai:gpt-5, simulated")
    c.add_argument("--rules-model", default=None, help="LLM that parses a .txt spec (defaults to --model)")
    c.add_argument("--px-per-m", type=float, default=None)
    c.add_argument("--no-snap", action="store_true")
    c.add_argument("--max-attempts", type=int, default=3)
    c.add_argument("--out", default="reports")

    e = sub.add_parser("eval", help="compare models on a labelled set")
    e.add_argument("--plans", default="data/plans")
    e.add_argument("--models", nargs="+", default=["simulated"])
    e.add_argument("--rules", default="data/rules/residential.yaml")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--prices", default="prices.yaml")
    e.add_argument("--no-snap", action="store_true")
    e.add_argument("--out", default="results")

    a = ap.parse_args(argv)
    if a.cmd == "synth":
        paths = make_dataset(a.out, a.n, a.seed, a.px_per_m)
        print(f"wrote {len(paths)} plans + ground truth to {a.out}")
    elif a.cmd == "check":
        meta = _meta(a.image, a.px_per_m)
        rules, text, rmodel = None, None, None
        if a.rules.endswith((".yaml", ".yml")):
            rules = load_rules_yaml(a.rules)
        else:
            text, rmodel = Path(a.rules).read_text(), a.rules_model or a.model
            if rmodel == "simulated":
                raise SystemExit("parsing a .txt spec needs a real LLM: pass --rules-model provider:model")
        state = run_plan(a.image, meta, make_extractor(a.model), rules=rules, rules_text=text,
                         rules_model=rmodel, max_attempts=a.max_attempts, use_snap=not a.no_snap)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(a.image).stem
        (out / f"{stem}.md").write_text(markdown(state))
        overlay(state, out / f"{stem}_overlay.png")
        (out / f"{stem}.json").write_text(json.dumps({
            "results": [r.model_dump() for r in state["results"]], "geometry": state["geometry"],
            "plan": state["plan"].model_dump(), "issues": state["issues"], "log": state["log"]}, indent=1))
        n_fail = sum(not r.passed for r in state["results"])
        print("\n".join(state["log"]))
        print(f"\n{len(state['results']) - n_fail}/{len(state['results'])} rules pass -> {out / (stem + '.md')}")
    elif a.cmd == "eval":
        summary = evaluate(a.plans, a.models, load_rules_yaml(a.rules), a.out, a.limit, a.prices,
                           use_snap=not a.no_snap)
        print("\n" + (Path(a.out) / "summary.md").read_text())


if __name__ == "__main__":
    main()
