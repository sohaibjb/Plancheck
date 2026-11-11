"""Turn a written specification into typed, checkable rules with an LLM (structured output)."""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from .extract import Usage
from .rules import RuleSet, RuleSetOut

PROMPT = """Convert this building specification into checkable rules.
Use exactly one rule per requirement, keep the original sentence in `text`, and number ids R01, R02, ...
Map room words to types: hall/passage/corridor -> corridor; lounge/living -> living; WC/bath/shower -> bathroom.
'any' means every room. Distances in metres, areas in m².
If a requirement cannot be expressed with the available rule kinds, leave it out.

SPECIFICATION:
{spec}"""


def load_rules_yaml(path: str | Path) -> RuleSet:
    return RuleSet.model_validate(yaml.safe_load(Path(path).read_text()))


def parse_rules_text(text: str, model: str) -> tuple[RuleSet, Usage]:
    from langchain.chat_models import init_chat_model

    llm = init_chat_model(model).with_structured_output(RuleSetOut, method="json_schema", include_raw=True)
    t0 = time.perf_counter()
    out = llm.invoke(PROMPT.format(spec=text))
    if out.get("parsed") is None:
        raise ValueError(f"rule parsing failed: {out.get('parsing_error')}")
    um = getattr(out["raw"], "usage_metadata", None) or {}
    return out["parsed"].to_ruleset(), Usage(um.get("input_tokens", 0), um.get("output_tokens", 0), time.perf_counter() - t0, 1)
