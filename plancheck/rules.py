"""Machine-checkable rules and their evaluation.

Rules are written by people in plain language (see data/rules/*.txt). The agent turns that text
into the typed rules below with structured output; each rule is then decided by exact geometry.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from .geometry import PlanGeometry
from .schema import RoomType

AnyType = Union[RoomType, Literal["any"]]


class _Rule(BaseModel):
    id: str = Field(description="Short id, e.g. 'R01'.")
    text: str = Field(description="The original sentence this rule comes from.")


class MinArea(_Rule):
    kind: Literal["min_area"] = "min_area"
    room_type: AnyType
    min_m2: float = Field(gt=0)


class MinWidth(_Rule):
    kind: Literal["min_width"] = "min_width"
    room_type: AnyType
    min_m: float = Field(gt=0)


class Adjacent(_Rule):
    """Every room of type `a` shares a wall with at least one room of type `b`."""
    kind: Literal["adjacent"] = "adjacent"
    a: RoomType
    b: RoomType


class DoorBetween(_Rule):
    """Every room of type `a` has a door directly into a room of type `b`."""
    kind: Literal["door_between"] = "door_between"
    a: RoomType
    b: RoomType


class NoDoorBetween(_Rule):
    """No door connects a room of type `a` directly to a room of type `b`."""
    kind: Literal["no_door_between"] = "no_door_between"
    a: RoomType
    b: RoomType


class MaxEgress(_Rule):
    """From the farthest point of every room of `room_type`, the walking distance to an exit is at most `max_m`."""
    kind: Literal["max_egress"] = "max_egress"
    room_type: AnyType = "any"
    max_m: float = Field(gt=0)


class EveryRoomHasDoor(_Rule):
    kind: Literal["every_room_has_door"] = "every_room_has_door"


class RequiresRoom(_Rule):
    kind: Literal["requires_room"] = "requires_room"
    room_type: RoomType
    min_count: int = Field(default=1, ge=1)


Rule = Annotated[
    Union[MinArea, MinWidth, Adjacent, DoorBetween, NoDoorBetween, MaxEgress, EveryRoomHasDoor, RequiresRoom],
    Field(discriminator="kind"),
]


class RuleSet(BaseModel):
    rules: list[Rule] = Field(description="One entry per checkable requirement in the text.")


class RuleResult(BaseModel):
    rule_id: str
    kind: str
    text: str
    passed: bool
    failures: list[str] = Field(default_factory=list, description="Human-readable reason per failing room.")
    failing_rooms: list[str] = Field(default_factory=list)


def _rooms_of(geo: PlanGeometry, t: str) -> list[str]:
    return [r for r, rt in geo.room_types.items() if (t == "any" or rt == t) and geo.rooms[r].is_valid]


def _name(geo: PlanGeometry, rid: str) -> str:
    return f"{geo.labels[rid]} ({rid})"


def check_rule(rule: Rule, geo: PlanGeometry) -> RuleResult:
    fails: list[tuple[str, str]] = []

    if isinstance(rule, MinArea):
        for r in _rooms_of(geo, rule.room_type):
            if geo.area(r) < rule.min_m2:
                fails.append((r, f"{_name(geo, r)} is {geo.area(r):.2f} m² (min {rule.min_m2} m²)"))

    elif isinstance(rule, MinWidth):
        for r in _rooms_of(geo, rule.room_type):
            if geo.min_width(r) < rule.min_m:
                fails.append((r, f"{_name(geo, r)} is {geo.min_width(r):.2f} m wide (min {rule.min_m} m)"))

    elif isinstance(rule, Adjacent):
        targets = _rooms_of(geo, rule.b)
        for r in _rooms_of(geo, rule.a):
            if not any(geo.adjacency.has_edge(r, t) for t in targets if t != r):
                fails.append((r, f"{_name(geo, r)} shares no wall with any {rule.b}"))

    elif isinstance(rule, DoorBetween):
        targets = _rooms_of(geo, rule.b)
        for r in _rooms_of(geo, rule.a):
            if not any(geo.door_between(r, t) for t in targets if t != r):
                fails.append((r, f"{_name(geo, r)} has no door into a {rule.b}"))

    elif isinstance(rule, NoDoorBetween):
        targets = _rooms_of(geo, rule.b)
        for r in _rooms_of(geo, rule.a):
            for t in targets:
                if t != r and geo.door_between(r, t):
                    fails.append((r, f"{_name(geo, r)} opens directly into {_name(geo, t)}"))

    elif isinstance(rule, MaxEgress):
        for r in _rooms_of(geo, rule.room_type):
            d, path = geo.egress[r]
            if math.isinf(d):
                fails.append((r, f"{_name(geo, r)} has no route to an exit"))
            elif d > rule.max_m:
                fails.append((r, f"{_name(geo, r)} is {d:.1f} m from an exit via {' → '.join(path)} (max {rule.max_m} m)"))

    elif isinstance(rule, EveryRoomHasDoor):
        for r in _rooms_of(geo, "any"):
            if not geo.doors_of(r):
                fails.append((r, f"{_name(geo, r)} has no door"))

    elif isinstance(rule, RequiresRoom):
        n = len(_rooms_of(geo, rule.room_type))
        if n < rule.min_count:
            fails.append(("", f"plan has {n} {rule.room_type} room(s) (needs {rule.min_count})"))

    return RuleResult(
        rule_id=rule.id, kind=rule.kind, text=rule.text, passed=not fails,
        failures=[m for _, m in fails], failing_rooms=sorted({r for r, _ in fails if r}),
    )


def check_all(rules: RuleSet, geo: PlanGeometry) -> list[RuleResult]:
    return [check_rule(r, geo) for r in rules.rules]


# ---------------------------------------------------------------------------
# Wire format for the rule parser: one flat object per rule (unused fields null), which every
# provider's structured-output mode accepts; converted to the typed rules above.
# ---------------------------------------------------------------------------

from pydantic import TypeAdapter  # noqa: E402

RuleKind = Literal["min_area", "min_width", "adjacent", "door_between", "no_door_between", "max_egress",
                   "every_room_has_door", "requires_room"]


class RuleOut(BaseModel):
    id: str = Field(description="R01, R02, ...")
    text: str = Field(description="The original sentence.")
    kind: RuleKind = Field(description=(
        "min_area(room_type, min_m2) | min_width(room_type, min_m) | adjacent(a, b): every a shares a wall with a b | "
        "door_between(a, b): every a has a door into a b | no_door_between(a, b) | "
        "max_egress(room_type, max_m): walking distance to an exit | every_room_has_door | "
        "requires_room(room_type, min_count)"))
    room_type: AnyType | None = None
    a: RoomType | None = None
    b: RoomType | None = None
    min_m2: float | None = None
    min_m: float | None = None
    max_m: float | None = None
    min_count: int | None = None


class RuleSetOut(BaseModel):
    """Checkable rules extracted from a written specification."""

    rules: list[RuleOut]

    def to_ruleset(self) -> RuleSet:
        ta = TypeAdapter(Rule)
        return RuleSet(rules=[ta.validate_python({k: v for k, v in r.model_dump().items() if v is not None})
                              for r in self.rules])
