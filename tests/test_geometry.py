import math

import pytest

from plancheck.geometry import PlanGeometry
from plancheck.schema import FloorPlan, Room

from .conftest import door, sq


def test_areas_and_widths(flat, meta):
    g = PlanGeometry(flat, meta)
    assert g.area("B") == pytest.approx(9.0)
    assert g.min_width("H") == pytest.approx(1.2)
    assert g.footprint.area == pytest.approx(60.0, rel=1e-3)


def test_adjacency_needs_a_real_shared_wall(flat, meta):
    g = PlanGeometry(flat, meta)
    assert g.adjacent("B", "T")          # 3 m shared wall
    assert g.adjacent("L", "K")          # 1.8 m shared wall
    assert not g.adjacent("B", "K")      # separated by the hall
    corner = FloorPlan(rooms=[Room(id="a", label="a", type="other", polygon=sq(0, 0, 2, 2)),
                              Room(id="b", label="b", type="other", polygon=sq(2, 2, 4, 4))], doors=[])
    assert not PlanGeometry(corner, meta).adjacent("a", "b")  # touching corners only


def test_doors_are_assigned_geometrically(flat, meta):
    g = PlanGeometry(flat, meta)
    assert set(g.door_rooms["D1"]) == {"B", "H"}
    assert g.exits == ["D5"]
    assert g.door_between("T", "H") == ["D2"]


def test_egress_follows_doors(flat, meta):
    g = PlanGeometry(flat, meta)
    dist, path = g.egress["T"]
    assert path == ["D2", "D3", "D5"]
    # farthest corner of the bath to D2, then D2->D3 across the hall, then D3->D5 across the living room
    expected = math.dist((10, 0), (9, 3)) + math.dist((9, 3), (5, 3.6)) + math.dist((5, 3.6), (1, 6))
    assert dist == pytest.approx(expected, rel=1e-6)


def test_geodesic_goes_around_corners(meta):
    # L-shaped corridor: straight line between the two ends leaves the room
    L = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]
    plan = FloorPlan(rooms=[Room(id="C", label="C", type="corridor", polygon=[(x * 100, y * 100) for x, y in L])],
                     doors=[])
    g = PlanGeometry(plan, meta)
    from shapely.geometry import Point
    d = g.geodesic("C", Point(3.5, 0.5), Point(0.5, 3.5))
    assert d > math.dist((3.5, 0.5), (0.5, 3.5))
    assert d == pytest.approx(2 * math.dist((3.5, 0.5), (1, 1)), rel=1e-6)


def test_no_exit_means_infinite_egress(flat, meta):
    flat.doors = [d for d in flat.doors if d.id != "D5"]
    g = PlanGeometry(flat, meta)
    assert all(math.isinf(d) for d, _ in g.egress.values())
    assert any(i.code == "no_exit" for i in g.validate())


def test_validation_finds_overlaps_gaps_and_floating_doors(flat, meta):
    assert PlanGeometry(flat, meta).errors() == []
    bad = flat.model_copy(deep=True)
    bad.rooms[1].polygon = sq(4.5, 0, 8, 3)  # bedroom pushed into the living room
    assert any(i.code == "overlap" for i in PlanGeometry(bad, meta).errors())
    ring = [sq(0, 0, 3, 1), sq(0, 1, 1, 2), sq(2, 1, 3, 2), sq(0, 2, 3, 3)]  # 1x1 m centre left empty
    gap = FloorPlan(rooms=[Room(id=f"r{i}", label="x", type="other", polygon=p) for i, p in enumerate(ring)],
                    doors=[])
    assert any(i.code == "gap" for i in PlanGeometry(gap, meta).errors())
    floating = flat.model_copy(deep=True)
    floating.doors.append(door(9, 2.5, 2.5))
    assert any(i.code == "floating_door" for i in PlanGeometry(floating, meta).errors())
