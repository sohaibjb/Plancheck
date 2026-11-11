"""Exact geometry over an extracted floor plan.

The model only *reads* the drawing. Everything measurable (areas, widths, which rooms share a
wall, where doors lead, how far each room is from an exit) is computed here, in metres, with
Shapely and NetworkX, so the verdicts are deterministic and auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations

import networkx as nx
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.validation import explain_validity

from .schema import FloorPlan, PlanMeta

EXIT = "__EXIT__"


@dataclass
class Issue:
    severity: str  # "error" -> plan must be re-extracted; "warning" -> report only
    code: str
    message: str
    subject: str | None = None


@dataclass
class Tolerances:
    wall_m: float = 0.25        # how far a door / shared edge may sit from a wall line
    min_shared_wall_m: float = 0.8  # shorter contact (e.g. touching corners) is not adjacency
    min_room_area_m2: float = 0.5
    max_overlap_m2: float = 0.25
    max_gap_m2: float = 0.5


@dataclass
class PlanGeometry:
    plan: FloorPlan
    meta: PlanMeta
    tol: Tolerances = field(default_factory=Tolerances)

    # ---------- shapes in metres ----------
    @cached_property
    def rooms(self) -> dict[str, Polygon]:
        s = 1.0 / self.meta.px_per_m
        out = {}
        for r in self.plan.rooms:
            poly = Polygon(r.polygon)
            out[r.id] = affinity.scale(poly, xfact=s, yfact=s, origin=(0, 0))
        return out

    @cached_property
    def room_types(self) -> dict[str, str]:
        return {r.id: r.type for r in self.plan.rooms}

    @cached_property
    def labels(self) -> dict[str, str]:
        return {r.id: r.label for r in self.plan.rooms}

    @cached_property
    def doors(self) -> dict[str, Point]:
        s = 1.0 / self.meta.px_per_m
        return {d.id: Point(d.position[0] * s, d.position[1] * s) for d in self.plan.doors}

    @cached_property
    def footprint(self) -> Polygon:
        eps = self.tol.wall_m / 2  # closes small gaps left by imprecise extraction
        valid = [p for p in self.rooms.values() if p.is_valid and not p.is_empty]
        u = unary_union([p.buffer(eps, join_style="mitre") for p in valid]).buffer(-eps, join_style="mitre")
        return u

    # ---------- per-room measures ----------
    def area(self, rid: str) -> float:
        return self.rooms[rid].area

    def min_width(self, rid: str) -> float:
        """Shorter side of the minimum rotated bounding rectangle (exact for rectangles)."""
        rect = self.rooms[rid].minimum_rotated_rectangle
        xs = list(rect.exterior.coords)
        sides = [math.dist(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
        return min(sides) if sides else 0.0

    # ---------- relations ----------
    def shared_wall_length(self, a: str, b: str) -> float:
        pa, pb = self.rooms[a], self.rooms[b]
        if pa.distance(pb) > self.tol.wall_m:
            return 0.0
        ba, bb = pa.exterior, pb.exterior
        l1 = bb.intersection(ba.buffer(self.tol.wall_m, cap_style="flat")).length
        l2 = ba.intersection(bb.buffer(self.tol.wall_m, cap_style="flat")).length
        return min(l1, l2)

    def adjacent(self, a: str, b: str) -> bool:
        return self.shared_wall_length(a, b) >= self.tol.min_shared_wall_m

    @cached_property
    def adjacency(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(self.rooms)
        for a, b in combinations(self.rooms, 2):
            L = self.shared_wall_length(a, b)
            if L >= self.tol.min_shared_wall_m:
                g.add_edge(a, b, length=L)
        return g

    @cached_property
    def door_rooms(self) -> dict[str, list[str]]:
        """Which rooms each door opens into (geometric, not taken from the model)."""
        out: dict[str, list[str]] = {}
        for did, p in self.doors.items():
            near = [(self.rooms[r].exterior.distance(p), r) for r in self.rooms]
            out[did] = [r for d, r in sorted(near) if d <= self.tol.wall_m][:2]
        return out

    @cached_property
    def outer_boundary(self):
        fp = self.footprint
        parts = list(fp.geoms) if hasattr(fp, "geoms") else [fp]
        return unary_union([q.exterior for q in parts if not q.is_empty])

    def is_exit(self, did: str) -> bool:
        p = self.doors[did]
        on_outer = self.outer_boundary.distance(p) <= self.tol.wall_m
        return on_outer and len(self.door_rooms[did]) == 1

    @cached_property
    def exits(self) -> list[str]:
        return [d for d in self.doors if self.is_exit(d)]

    def doors_of(self, rid: str) -> list[str]:
        return [d for d, rs in self.door_rooms.items() if rid in rs]

    def door_between(self, a: str, b: str) -> list[str]:
        return [d for d, rs in self.door_rooms.items() if a in rs and b in rs]

    # ---------- egress ----------
    def _on_boundary(self, rid: str, did: str) -> Point:
        poly = self.rooms[rid]
        return nearest_points(poly.exterior, self.doors[did])[0]

    def geodesic(self, rid: str, p: Point, q: Point) -> float:
        """Shortest path from p to q staying inside room `rid` (exact for simple polygons)."""
        poly = self.rooms[rid].buffer(1e-6)
        if poly.covers(LineString([p, q])):
            return p.distance(q)
        nodes = [p, q] + [Point(c) for c in self.rooms[rid].exterior.coords[:-1]]
        g = nx.Graph()
        for i, j in combinations(range(len(nodes)), 2):
            seg = LineString([nodes[i], nodes[j]])
            if poly.covers(seg):
                g.add_edge(i, j, w=seg.length)
        try:
            return nx.shortest_path_length(g, 0, 1, weight="w")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return math.inf

    @cached_property
    def egress_graph(self) -> nx.Graph:
        """Nodes are doors (+ EXIT); an edge crosses one room from door to door."""
        g = nx.Graph()
        g.add_node(EXIT)
        for rid in self.rooms:
            ds = self.doors_of(rid)
            for d1, d2 in combinations(ds, 2):
                w = self.geodesic(rid, self._on_boundary(rid, d1), self._on_boundary(rid, d2))
                if g.has_edge(d1, d2):
                    w = min(w, g[d1][d2]["w"])
                g.add_edge(d1, d2, w=w, via=rid)
        for e in self.exits:
            g.add_edge(e, EXIT, w=0.0, via=None)
        return g

    @cached_property
    def egress(self) -> dict[str, tuple[float, list[str]]]:
        """Worst-case travel distance (m) from the farthest point of each room to an exit, and the door path."""
        g = self.egress_graph
        dist, paths = nx.single_source_dijkstra(g, EXIT, weight="w")
        out: dict[str, tuple[float, list[str]]] = {}
        for rid, poly in self.rooms.items():
            best = (math.inf, [])
            corners = [Point(c) for c in poly.exterior.coords[:-1]]
            for d in self.doors_of(rid):
                if d not in dist:
                    continue
                dp = self._on_boundary(rid, d)
                inside = max(self.geodesic(rid, c, dp) for c in corners)
                total = inside + dist[d]
                if total < best[0]:
                    best = (total, list(reversed(paths[d]))[:-1])
            out[rid] = best
        return out

    # ---------- validation (drives the agent's repair loop) ----------
    def validate(self) -> list[Issue]:
        issues: list[Issue] = []
        ids = [r.id for r in self.plan.rooms]
        for dup in {i for i in ids if ids.count(i) > 1}:
            issues.append(Issue("error", "duplicate_id", f"Room id {dup} is used more than once.", dup))
        if not self.plan.rooms:
            issues.append(Issue("error", "no_rooms", "No rooms were extracted."))
            return issues
        for rid, poly in self.rooms.items():
            if not poly.is_valid:
                issues.append(Issue("error", "invalid_polygon",
                                    f"Room {rid} polygon is invalid ({explain_validity(poly)}).", rid))
            elif poly.area < self.tol.min_room_area_m2:
                issues.append(Issue("error", "tiny_room",
                                    f"Room {rid} has area {poly.area:.2f} m², which is too small to be a room.", rid))
        valid = {k: v for k, v in self.rooms.items() if v.is_valid}
        for a, b in combinations(valid, 2):
            ov = valid[a].intersection(valid[b]).area
            if ov > self.tol.max_overlap_m2:
                issues.append(Issue("error", "overlap",
                                    f"Rooms {a} and {b} overlap by {ov:.2f} m²; polygons must follow wall centrelines.",
                                    f"{a},{b}"))
        fp = self.footprint
        holes = list(fp.interiors) if isinstance(fp, Polygon) else []
        for h in holes:
            if Polygon(h).area > self.tol.max_gap_m2:
                c = Polygon(h).centroid
                issues.append(Issue("error", "gap",
                                    f"Unassigned gap of {Polygon(h).area:.2f} m² near ({c.x * self.meta.px_per_m:.0f}, "
                                    f"{c.y * self.meta.px_per_m:.0f}) px: a room may be missing."))
        if not isinstance(fp, Polygon):
            issues.append(Issue("warning", "disconnected", "Rooms form more than one separate block."))
        for did, rs in self.door_rooms.items():
            if not rs:
                issues.append(Issue("error", "floating_door",
                                    f"Door {did} is not on any room wall.", did))
        for rid in valid:
            if not self.doors_of(rid):
                issues.append(Issue("warning", "no_door", f"Room {rid} has no door.", rid))
        if self.plan.doors and not self.exits:
            issues.append(Issue("warning", "no_exit", "No door leads outside the building."))
        return issues

    def errors(self) -> list[Issue]:
        return [i for i in self.validate() if i.severity == "error"]

    def summary(self) -> dict:
        return {
            "rooms": [
                {
                    "id": rid,
                    "label": self.labels[rid],
                    "type": self.room_types[rid],
                    "area_m2": round(self.area(rid), 2),
                    "min_width_m": round(self.min_width(rid), 2),
                    "doors": self.doors_of(rid),
                    "egress_m": None if math.isinf(self.egress[rid][0]) else round(self.egress[rid][0], 2),
                }
                for rid in self.rooms
                if self.rooms[rid].is_valid
            ],
            "adjacency": sorted([sorted([a, b]) for a, b in self.adjacency.edges]),
            "exits": self.exits,
        }
