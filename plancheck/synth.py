"""Synthetic floor plans with exact ground truth.

Each plan is a random apartment (living room, kitchen, bathroom, bedrooms around a corridor),
drawn as a high-resolution architectural-style PNG, with the true rooms/doors saved as JSON.
Some plans deliberately break typical rules (small bedrooms, bathroom opening into the kitchen,
long escape routes) so the test set has both passing and failing cases.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

from .schema import Door, FloorPlan, PlanMeta, Room

WALL_M = 0.15
DOOR_M = 0.9
MARGIN_M = 1.6


@dataclass
class _R:
    x0: float
    y0: float
    x1: float
    y1: float
    type: str
    label: str = ""


def _font(size: int) -> ImageFont.FreeTypeFont:
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return ImageFont.truetype(str(path), size)


def _split(x0: float, x1: float, n: int, rng: random.Random, min_w: float) -> list[float]:
    """Split [x0, x1] into n parts, each at least min_w, snapped to 5 cm."""
    for _ in range(200):
        cuts = sorted(rng.uniform(x0, x1) for _ in range(n - 1))
        xs = [x0] + [round(c * 20) / 20 for c in cuts] + [x1]
        if all(b - a >= min_w for a, b in zip(xs, xs[1:])):
            return xs
    step = (x1 - x0) / n
    return [round((x0 + i * step) * 20) / 20 for i in range(n)] + [x1]


def generate_layout(rng: random.Random) -> tuple[list[_R], list[tuple[float, float]]]:
    """Return rooms (metres) and door centre points (metres)."""
    W = round(rng.uniform(11, 17) * 2) / 2
    H = round(rng.uniform(8, 11) * 2) / 2
    living_w = round(rng.uniform(3.6, 5.0) * 2) / 2
    cor_h = rng.choice([1.1, 1.2, 1.3])
    top_h = round(rng.uniform(0.42, 0.55) * (H - cor_h) * 10) / 10
    cy0, cy1 = top_h, top_h + cor_h

    rooms: list[_R] = [_R(0, 0, living_w, H, "living"), _R(living_w, cy0, W, cy1, "corridor")]
    small_bed = rng.random() < 0.35  # sometimes create an undersized bedroom
    min_w = 2.2 if small_bed else 2.8
    n_top = 2 if W - living_w < 9 else 3
    n_bot = 2 if W - living_w < 8 else 3
    xt = _split(living_w, W, n_top, rng, min_w)
    xb = _split(living_w, W, n_bot, rng, min_w)
    top = [_R(a, 0, b, cy0, "bedroom") for a, b in zip(xt, xt[1:])]
    bot = [_R(a, cy1, b, H, "bedroom") for a, b in zip(xb, xb[1:])]
    top[0].type = "kitchen"
    # bathroom: either next to the kitchen (top row) or in the bottom row
    bath_next_to_kitchen = rng.random() < 0.4 and len(top) > 2
    if bath_next_to_kitchen:
        top[1].type = "bathroom"
    else:
        min(bot, key=lambda r: r.x1 - r.x0).type = "bathroom"
    rooms += top + bot

    doors: list[tuple[float, float]] = []

    def door_on_h_wall(r: _R, y: float) -> tuple[float, float]:
        lo, hi = r.x0 + 0.6, r.x1 - 0.6
        return (round(rng.uniform(lo, hi), 2), y)

    for r in top:
        doors.append(door_on_h_wall(r, cy0))
    for r in bot:
        doors.append(door_on_h_wall(r, cy1))
    doors.append((living_w, (cy0 + cy1) / 2))  # living <-> corridor
    doors.append((0.0, round(rng.uniform(1.0, H - 1.0), 2)))  # front door on living exterior wall
    if rng.random() < 0.5:  # optional back exit at the corridor's far end
        doors.append((W, (cy0 + cy1) / 2))
    if rng.random() < 0.5:  # kitchen also opens to the living room
        k = top[0]
        doors.append((living_w, round(rng.uniform(k.y0 + 0.6, k.y1 - 0.6), 2)))
    if bath_next_to_kitchen and rng.random() < 0.6:  # bad practice: bathroom opens into kitchen
        k, b = top[0], top[1]
        doors.append((k.x1, round(rng.uniform(0.7, k.y1 - 0.7), 2)))

    # labels
    n_bed = 0
    for r in rooms:
        if r.type == "bedroom":
            n_bed += 1
            r.label = f"BEDROOM {n_bed}"
        else:
            r.label = {"living": "LIVING ROOM", "kitchen": "KITCHEN", "bathroom": "BATH",
                       "corridor": "HALL"}[r.type]
    return rooms, doors


def render(rooms: list[_R], doors: list[tuple[float, float]], ppm: float, seed: int) -> tuple[Image.Image, FloorPlan, PlanMeta]:
    W = max(r.x1 for r in rooms)
    H = max(r.y1 for r in rooms)
    off = MARGIN_M
    size = (int((W + 2 * off) * ppm), int((H + 2 * off + 1.2) * ppm))
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)

    def px(x: float, y: float) -> tuple[float, float]:
        return ((x + off) * ppm, (y + off) * ppm)

    half = WALL_M * ppm / 2
    # walls: draw every room outline as thick rectangles
    for r in rooms:
        for (ax, ay), (bx, by) in [((r.x0, r.y0), (r.x1, r.y0)), ((r.x1, r.y0), (r.x1, r.y1)),
                                   ((r.x1, r.y1), (r.x0, r.y1)), ((r.x0, r.y1), (r.x0, r.y0))]:
            (pax, pay), (pbx, pby) = px(ax, ay), px(bx, by)
            d.rectangle([min(pax, pbx) - half, min(pay, pby) - half, max(pax, pbx) + half, max(pay, pby) + half],
                        fill="black")
    # door openings + swing arcs
    dw = DOOR_M * ppm
    for (x, y) in doors:
        cx, cy = px(x, y)
        on_vertical = any(abs(x - v) < 1e-6 for r in rooms for v in (r.x0, r.x1)) and not any(
            abs(y - h) < 1e-6 for r in rooms for h in (r.y0, r.y1))
        if on_vertical:
            d.rectangle([cx - half - 1, cy - dw / 2, cx + half + 1, cy + dw / 2], fill="white")
            d.line([cx, cy - dw / 2, cx + dw, cy - dw / 2], fill=(90, 90, 90), width=2)
            d.arc([cx - dw, cy - dw / 2 - dw, cx + dw, cy - dw / 2 + dw], 0, 90, fill=(140, 140, 140), width=2)
        else:
            d.rectangle([cx - dw / 2, cy - half - 1, cx + dw / 2, cy + half + 1], fill="white")
            d.line([cx - dw / 2, cy, cx - dw / 2, cy + dw], fill=(90, 90, 90), width=2)
            d.arc([cx - dw / 2 - dw, cy - dw, cx - dw / 2 + dw, cy + dw], 0, 90, fill=(140, 140, 140), width=2)
    # labels
    f = _font(int(ppm * 0.28))
    for r in rooms:
        cx, cy = px((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
        d.text((cx, cy), r.label, fill="black", font=f, anchor="mm")
    # scale bar + title block
    f2 = _font(int(ppm * 0.22))
    bx, by = px(0, H + 0.8)
    for i in range(5):
        d.rectangle([bx + i * ppm, by, bx + (i + 1) * ppm, by + 0.12 * ppm], fill="black" if i % 2 == 0 else "white",
                    outline="black", width=2)
        d.text((bx + i * ppm, by - 6), str(i), fill="black", font=f2, anchor="mb")
    d.text((bx + 5 * ppm, by - 6), "5 m", fill="black", font=f2, anchor="mb")
    d.text((size[0] - off * ppm, by + 0.06 * ppm), f"PLAN {seed:04d}   SCALE 1:100", fill="black", font=f2, anchor="rm")

    plan = FloorPlan(
        rooms=[Room(id=f"R{i + 1}", label=r.label, type=r.type,  # type: ignore[arg-type]
                    polygon=[px(r.x0, r.y0), px(r.x1, r.y0), px(r.x1, r.y1), px(r.x0, r.y1)])
               for i, r in enumerate(rooms)],
        doors=[Door(id=f"D{i + 1}", position=px(x, y), width_px=dw) for i, (x, y) in enumerate(doors)],
    )
    return img, plan, PlanMeta(px_per_m=ppm, image_size=size)


def make_dataset(out_dir: str | Path, n: int = 30, seed: int = 0, ppm: float = 120.0) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        s = seed * 10_000 + i
        rng = random.Random(s)
        rooms, doors = generate_layout(rng)
        img, plan, meta = render(rooms, doors, ppm, s)
        stem = out / f"plan_{s:04d}"
        img.save(stem.with_suffix(".png"))
        stem.with_suffix(".json").write_text(json.dumps(
            {"meta": meta.model_dump(), "plan": plan.model_dump()}, indent=1))
        paths.append(stem.with_suffix(".png"))
    return paths


def load_truth(png_path: str | Path) -> tuple[FloorPlan, PlanMeta]:
    data = json.loads(Path(png_path).with_suffix(".json").read_text())
    return FloorPlan.model_validate(data["plan"]), PlanMeta.model_validate(data["meta"])
