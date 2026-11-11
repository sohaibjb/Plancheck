import random

import pytest
from PIL import Image

from plancheck.agent import run_plan
from plancheck.extract import SimulatedExtractor, _scale_plan
from plancheck.geometry import PlanGeometry
from plancheck.parse_rules import load_rules_yaml
from plancheck.rules import check_all
from plancheck.schema import FloorPlan, Room
from plancheck.snap import snap_plan, uncovered_regions
from plancheck.synth import load_truth, make_dataset


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("plans"), n=6, seed=7)


@pytest.fixture(scope="module")
def rules():
    return load_rules_yaml("data/rules/residential.yaml")


def test_synthetic_truth_is_clean_and_deterministic(plans, tmp_path):
    for p in plans:
        plan, meta = load_truth(p)
        assert PlanGeometry(plan, meta).errors() == []
    again = make_dataset(tmp_path, n=1, seed=7)[0]
    assert again.with_suffix(".json").read_text() == plans[0].with_suffix(".json").read_text()


def test_snapping_recovers_exact_walls(plans):
    plan, meta = load_truth(plans[0])
    rng = random.Random(1)
    noisy = FloorPlan(rooms=[Room(id=r.id, label=r.label, type=r.type,
                                  polygon=[(x + rng.gauss(0, 15), y + rng.gauss(0, 15)) for x, y in r.polygon])
                             for r in plan.rooms], doors=plan.doors)
    snapped, _ = snap_plan(noisy, Image.open(plans[0]), meta.px_per_m)
    for t, s in zip(plan.rooms, snapped.rooms):
        for (tx, ty), (sx, sy) in zip(t.polygon, s.polygon):
            assert abs(tx - sx) <= 2 and abs(ty - sy) <= 2


def test_missing_room_is_detected_from_the_image(plans):
    plan, meta = load_truth(plans[1])
    img = Image.open(plans[1])
    assert uncovered_regions(plan, img, meta.px_per_m) == []
    dropped = FloorPlan(rooms=plan.rooms[:-1], doors=plan.doors)  # an outer-edge room
    found = uncovered_regions(dropped, img, meta.px_per_m)
    assert len(found) == 1 and found[0]["area_m2"] > 4


def test_agent_repairs_and_matches_ground_truth(plans, rules):
    for p in plans:
        plan, meta = load_truth(p)
        truth = {r.rule_id: r.passed for r in check_all(rules, PlanGeometry(plan, meta))}
        state = run_plan(str(p), meta, SimulatedExtractor(p_drop_room=1.0, p_wrong_type=0), rules=rules)
        assert state["attempts"] == 2  # first answer missed a room, the repair fixed it
        assert {r.rule_id: r.passed for r in state["results"]} == truth


def test_scale_roundtrip(plans):
    plan, _ = load_truth(plans[0])
    back = _scale_plan(_scale_plan(plan, 0.5), 2.0)
    assert back.rooms[0].polygon == pytest.approx(plan.rooms[0].polygon)


def test_vlm_extractor_path_with_stub_model(plans, rules):
    """Exercise VLMExtractor end to end with a stand-in for the chat model (no network)."""
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    from plancheck.extract import VLMExtractor, _image_payload
    from plancheck.schema import FloorPlanOut

    plan, meta = load_truth(plans[2])
    _, s, _ = _image_payload(str(plans[2]), 1568)
    seen = {}

    def fake_model(messages):
        seen["text"] = messages[1].content[0]["text"]
        seen["image"] = messages[1].content[1]["image_url"]["url"][:22]
        # a model answers in the *downscaled* image's pixels
        return {"raw": AIMessage(content="", usage_metadata={"input_tokens": 1500, "output_tokens": 700,
                                                             "total_tokens": 2200}),
                "parsed": FloorPlanOut.from_plan(_scale_plan(plan, s)), "parsing_error": None}

    ex = VLMExtractor.__new__(VLMExtractor)
    ex.model, ex.max_side, ex.temperature, ex.name = "stub", 1568, None, "stub"
    ex._llm = RunnableLambda(fake_model)
    state = run_plan(str(plans[2]), meta, ex, rules=rules)
    assert seen["image"] == "data:image/png;base64,"
    assert "pixels" in seen["text"]
    truth = {r.rule_id: r.passed for r in check_all(rules, PlanGeometry(plan, meta))}
    assert {r.rule_id: r.passed for r in state["results"]} == truth
    assert state["usage"][0]["input_tokens"] == 1500
