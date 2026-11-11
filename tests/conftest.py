import pytest

from plancheck.schema import Door, FloorPlan, PlanMeta, Room

PPM = 100.0  # 1 m = 100 px in hand-made plans


def sq(x0, y0, x1, y1):
    return [(x0 * PPM, y0 * PPM), (x1 * PPM, y0 * PPM), (x1 * PPM, y1 * PPM), (x0 * PPM, y1 * PPM)]


def door(i, x, y):
    return Door(id=f"D{i}", position=(x * PPM, y * PPM))


@pytest.fixture
def meta():
    return PlanMeta(px_per_m=PPM)


@pytest.fixture
def flat():
    """Living 5x6 on the left, hall 1.2 wide, bedroom 3x3 (9 m²) and bath 2x3 above, kitchen below.

    y=0 ┌──────────┬─────────┬──────┐
        │          │ BED 3x3 │ BATH │
    y=3 │  LIVING  ├─D1──────┴─D2───┤
        │          D3     HALL      │
    4.2 │          ├─D4─────────────┤
        │          │   KITCHEN      │
    y=6 └D5────────┴────────────────┘
       x=0        x=5      x=8     x=10
    """
    rooms = [
        Room(id="L", label="LIVING", type="living", polygon=sq(0, 0, 5, 6)),
        Room(id="B", label="BEDROOM", type="bedroom", polygon=sq(5, 0, 8, 3)),
        Room(id="T", label="BATH", type="bathroom", polygon=sq(8, 0, 10, 3)),
        Room(id="H", label="HALL", type="corridor", polygon=sq(5, 3, 10, 4.2)),
        Room(id="K", label="KITCHEN", type="kitchen", polygon=sq(5, 4.2, 10, 6)),
    ]
    doors = [door(1, 6.5, 3), door(2, 9, 3), door(3, 5, 3.6), door(4, 7, 4.2), door(5, 1, 6)]
    return FloorPlan(rooms=rooms, doors=doors)
