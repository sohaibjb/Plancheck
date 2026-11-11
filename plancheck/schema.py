"""Data model shared by the extractor, the geometry layer and the evaluator.

Coordinates are in pixels of the plan image (origin top-left, y down). The plan's scale
(`px_per_m`) converts them to metres. Room polygons follow wall *centrelines*, so two rooms
that share a wall share an edge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RoomType = Literal[
    "living", "kitchen", "bedroom", "bathroom", "corridor", "dining", "storage", "other"
]
ROOM_TYPES: tuple[str, ...] = RoomType.__args__  # type: ignore[attr-defined]


class Room(BaseModel):
    id: str = Field(description="Short unique id, e.g. 'R1'.")
    label: str = Field(description="Text written in the room on the drawing, e.g. 'BEDROOM 2'.")
    type: RoomType = Field(description="Room category.")
    polygon: list[tuple[float, float]] = Field(
        description="Room outline as [x, y] pixel points along the wall centrelines, in order, "
        "without repeating the first point. At least 3 points."
    )

    @field_validator("polygon")
    @classmethod
    def _min_points(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("a room polygon needs at least 3 points")
        return v


class Door(BaseModel):
    id: str = Field(description="Short unique id, e.g. 'D1'.")
    position: tuple[float, float] = Field(
        description="[x, y] pixel position of the centre of the door opening, on the wall line."
    )
    width_px: float | None = Field(default=None, description="Opening width in pixels, if visible.")


class FloorPlan(BaseModel):
    """Everything the checker needs from one drawing."""

    rooms: list[Room] = Field(description="Every enclosed room, corridor and hall.")
    doors: list[Door] = Field(
        description="Every door opening, including doors to the outside (building exits)."
    )


class PlanMeta(BaseModel):
    """Information about the drawing that is not extracted by the model."""

    px_per_m: float = Field(gt=0, description="Pixels per metre on the drawing.")
    image_size: tuple[int, int] | None = None


# ---------------------------------------------------------------------------
# Wire format: what the model is asked to return. Plain objects only (no tuples, unions or
# length constraints), so the same schema works with every provider's structured-output mode.
# ---------------------------------------------------------------------------

class XY(BaseModel):
    x: float
    y: float


class RoomOut(BaseModel):
    id: str = Field(description="Short unique id, e.g. 'R1'.")
    label: str = Field(description="Text written in the room on the drawing.")
    type: RoomType
    polygon: list[XY] = Field(description="Corners along the wall centrelines, in order, first point not repeated.")


class DoorOut(BaseModel):
    id: str = Field(description="Short unique id, e.g. 'D1'.")
    x: float = Field(description="x of the centre of the door opening, on the wall line.")
    y: float = Field(description="y of the centre of the door opening, on the wall line.")


class FloorPlanOut(BaseModel):
    """Rooms and doors read from a floor plan image (pixel coordinates)."""

    rooms: list[RoomOut]
    doors: list[DoorOut]

    def to_plan(self) -> FloorPlan:
        return FloorPlan(
            rooms=[Room(id=r.id, label=r.label, type=r.type, polygon=[(p.x, p.y) for p in r.polygon])
                   for r in self.rooms if len(r.polygon) >= 3],
            doors=[Door(id=d.id, position=(d.x, d.y)) for d in self.doors],
        )

    @classmethod
    def from_plan(cls, plan: FloorPlan) -> "FloorPlanOut":
        return cls(
            rooms=[RoomOut(id=r.id, label=r.label, type=r.type, polygon=[XY(x=x, y=y) for x, y in r.polygon])
                   for r in plan.rooms],
            doors=[DoorOut(id=d.id, x=d.position[0], y=d.position[1]) for d in plan.doors],
        )
