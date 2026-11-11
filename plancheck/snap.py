"""Snap model coordinates onto the walls actually drawn in the image.

Vision-language models see a downscaled image and are only roughly right about coordinates.
Walls in a plan are long, thick, dark strokes, so their centrelines can be found exactly from the
full-resolution pixels. We keep the model's *topology* (which rooms and doors exist) and replace
its coordinates with the nearest detected wall line and door gap.

Works for axis-aligned walls (most plans); angled walls are left as the model gave them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .schema import Door, FloorPlan, Room


@dataclass
class WallLines:
    xs: np.ndarray  # x of vertical wall centrelines (px)
    ys: np.ndarray  # y of horizontal wall centrelines (px)
    mask: np.ndarray  # True where a wall pixel is


def _longest_runs(mask: np.ndarray) -> np.ndarray:
    """Longest run of True along axis 1, per row."""
    m = np.pad(mask.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    out = np.zeros(mask.shape[0], dtype=int)
    for i in range(mask.shape[0]):
        starts = np.flatnonzero(d[i] == 1)
        ends = np.flatnonzero(d[i] == -1)
        if len(starts):
            out[i] = int((ends - starts).max())
    return out


def _bands(flags: np.ndarray, min_thick: int) -> np.ndarray:
    idx = np.flatnonzero(flags)
    if not len(idx):
        return np.array([])
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return np.array([(g[0] + g[-1]) / 2 for g in groups if len(g) >= min_thick])


def detect_walls(img: Image.Image, px_per_m: float, dark: int = 80, min_len_m: float = 0.8,
                 min_thick_m: float = 0.06) -> WallLines:
    gray = np.asarray(img.convert("L"))
    mask = gray < dark
    min_len = int(min_len_m * px_per_m)
    min_thick = max(2, int(min_thick_m * px_per_m))
    ys = _bands(_longest_runs(mask) >= min_len, min_thick)
    xs = _bands(_longest_runs(mask.T) >= min_len, min_thick)
    return WallLines(xs=xs, ys=ys, mask=mask)


def _nearest(v: float, cands: np.ndarray, tol: float) -> float | None:
    if not len(cands):
        return None
    i = int(np.argmin(np.abs(cands - v)))
    return float(cands[i]) if abs(cands[i] - v) <= tol else None


def _gap_centre(line: np.ndarray, pos: float, min_gap: int, max_gap: int, tol: float) -> float | None:
    """Centre of the white gap (door opening) in a wall pixel line, nearest to pos."""
    m = np.pad(line.astype(np.int8), 1, constant_values=1)
    d = np.diff(m)
    starts = np.flatnonzero(d == -1)  # dark -> white
    ends = np.flatnonzero(d == 1)
    best = None
    for s, e in zip(starts, ends):
        w = e - s
        if min_gap <= w <= max_gap:
            c = (s + e - 1) / 2
            if abs(c - pos) <= tol and (best is None or abs(c - pos) < abs(best - pos)):
                best = c
    return best


def _coverage(mask: np.ndarray, vertical: bool, at: float, lo: float, hi: float) -> float:
    """Share of wall pixels along a candidate wall line between lo and hi."""
    h, w = mask.shape
    a, b = int(max(0, min(lo, hi))), int(min((h if vertical else w) - 1, max(lo, hi)))
    if b <= a:
        return 0.0
    i = int(round(at))
    if vertical:
        seg = mask[a:b, max(0, min(w - 1, i))]
    else:
        seg = mask[max(0, min(h - 1, i)), a:b]
    return float(seg.mean())


def _best_line(mask, cands, vertical, at, lo, hi, tol):
    near = [c for c in cands if abs(c - at) <= tol]
    if not near:
        return None
    # prefer the line that is actually drawn along this edge; break ties by distance
    return max(near, key=lambda c: (round(_coverage(mask, vertical, c, lo, hi), 1), -abs(c - at)))


def snap_plan(plan: FloorPlan, img: Image.Image, px_per_m: float, tol_m: float = 0.45) -> tuple[FloorPlan, dict]:
    walls = detect_walls(img, px_per_m)
    tol = tol_m * px_per_m
    moved = []

    rooms = []
    for r in plan.rooms:
        pts = [list(p) for p in r.polygon]
        n = len(pts)
        new = [list(p) for p in pts]
        for i in range(n):  # snap edge by edge, so both ends of an edge land on the same wall
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
            if abs(x1 - x0) <= abs(y1 - y0):  # vertical-ish edge -> choose an x
                c = _best_line(walls.mask, walls.xs, True, (x0 + x1) / 2, y0, y1, tol)
                if c is not None:
                    new[i][0] = new[(i + 1) % n][0] = c
            else:
                c = _best_line(walls.mask, walls.ys, False, (y0 + y1) / 2, x0, x1, tol)
                if c is not None:
                    new[i][1] = new[(i + 1) % n][1] = c
        out = []
        for (ox, oy), (nx_, ny_) in zip(pts, new):
            moved.append(((nx_ - ox) ** 2 + (ny_ - oy) ** 2) ** 0.5)
            if not out or abs(out[-1][0] - nx_) > 1 or abs(out[-1][1] - ny_) > 1:
                out.append((nx_, ny_))
        if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= 1 and abs(out[0][1] - out[-1][1]) <= 1:
            out.pop()
        rooms.append(Room(id=r.id, label=r.label, type=r.type, polygon=out if len(out) >= 3 else r.polygon))

    doors = []
    h, w = walls.mask.shape
    gmin, gmax = int(0.6 * px_per_m), int(1.3 * px_per_m)
    for d in plan.doors:
        x, y = d.position
        options = []  # (perpendicular shift, new x, new y): a wall line that has a door-sized gap here
        for cy in walls.ys:
            if abs(cy - y) <= tol:
                g = _gap_centre(walls.mask[int(round(cy)), :], x, gmin, gmax, tol * 2)
                if g is not None:
                    options.append((abs(cy - y) + abs(g - x), g, float(cy)))
        for cx in walls.xs:
            if abs(cx - x) <= tol:
                g = _gap_centre(walls.mask[:, int(min(w - 1, round(cx)))], y, gmin, gmax, tol * 2)
                if g is not None:
                    options.append((abs(cx - x) + abs(g - y), float(cx), g))
        nx_, ny_ = (min(options)[1], min(options)[2]) if options else (x, y)
        doors.append(Door(id=d.id, position=(nx_, ny_), width_px=d.width_px))

    stats = {
        "wall_lines": {"vertical": len(walls.xs), "horizontal": len(walls.ys)},
        "mean_vertex_shift_px": round(float(np.mean(moved)) if moved else 0.0, 1),
    }
    return FloorPlan(rooms=rooms, doors=doors), stats


def uncovered_regions(plan: FloorPlan, img: Image.Image, px_per_m: float, min_area_m2: float = 1.0,
                      cell_m: float = 0.05) -> list[dict]:
    """Areas inside the drawn building outline that no extracted room covers.

    Catches rooms the model skipped, including rooms on the outer edge, which leave no
    gap *between* extracted rooms and so are invisible to polygon-only validation.
    """
    from PIL import ImageDraw
    from scipy import ndimage

    k = max(1, int(round(cell_m * px_per_m)))  # work on a coarse grid (~5 cm cells)
    mask = np.asarray(img.convert("L"))[::k, ::k] < 80
    door = max(3, int(round(1.3 * px_per_m / k)))
    closed = ndimage.binary_closing(mask, structure=np.ones((door, door)), border_value=0)
    inside = ndimage.binary_fill_holes(closed)
    inside = ndimage.binary_erosion(inside, iterations=max(1, door // 2))  # drop the outer wall band

    canvas = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
    dr = ImageDraw.Draw(canvas)
    for r in plan.rooms:
        if len(r.polygon) >= 3:
            dr.polygon([(x / k, y / k) for x, y in r.polygon], fill=1)
    covered = ndimage.binary_dilation(np.asarray(canvas, dtype=bool), iterations=max(1, int(0.2 * px_per_m / k)))

    missing = inside & ~covered & ~mask
    labels, n = ndimage.label(missing)
    out = []
    cell_area = (k / px_per_m) ** 2
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        area = len(xs) * cell_area
        if area >= min_area_m2:
            out.append({"x": float(xs.mean() * k), "y": float(ys.mean() * k), "area_m2": round(area, 1)})
    return out
