from plancheck.geometry import PlanGeometry
from plancheck.parse_rules import load_rules_yaml
from plancheck.rules import (Adjacent, DoorBetween, EveryRoomHasDoor, MaxEgress, MinArea, MinWidth, NoDoorBetween,
                             RequiresRoom, RuleSet, check_all, check_rule)

from .conftest import door


def test_each_rule_kind(flat, meta):
    g = PlanGeometry(flat, meta)
    ok = lambda r: check_rule(r, g).passed  # noqa: E731
    assert ok(MinArea(id="a", text="", room_type="bedroom", min_m2=9))
    assert not ok(MinArea(id="a", text="", room_type="bedroom", min_m2=9.5))
    assert ok(MinWidth(id="b", text="", room_type="bedroom", min_m=3))
    assert not ok(MinWidth(id="b", text="", room_type="corridor", min_m=1.5))
    assert ok(Adjacent(id="c", text="", a="kitchen", b="living"))
    assert not ok(Adjacent(id="c", text="", a="bedroom", b="kitchen"))
    assert ok(DoorBetween(id="d", text="", a="bedroom", b="corridor"))
    assert not ok(DoorBetween(id="d", text="", a="kitchen", b="living"))
    assert ok(NoDoorBetween(id="e", text="", a="bathroom", b="kitchen"))
    assert ok(EveryRoomHasDoor(id="f", text=""))
    assert ok(RequiresRoom(id="g", text="", room_type="bathroom"))
    assert not ok(RequiresRoom(id="g", text="", room_type="bathroom", min_count=2))
    assert ok(MaxEgress(id="h", text="", max_m=15))
    assert not ok(MaxEgress(id="h", text="", max_m=8))


def test_failures_name_the_rooms(flat, meta):
    flat.doors.append(door(9, 8, 1.5))  # bath now opens into the bedroom; add a bath->kitchen door instead
    flat.doors.append(door(10, 9.5, 4.2))
    res = check_rule(NoDoorBetween(id="x", text="", a="corridor", b="kitchen"), PlanGeometry(flat, meta))
    assert not res.passed and res.failing_rooms == ["H"]
    assert "KITCHEN" in res.failures[0]


def test_yaml_spec_loads_and_runs(flat, meta):
    rs = load_rules_yaml("data/rules/residential.yaml")
    assert isinstance(rs, RuleSet) and len(rs.rules) == 8
    res = check_all(rs, PlanGeometry(flat, meta))
    assert {r.rule_id for r in res} == {f"R0{i}" for i in range(1, 9)}
