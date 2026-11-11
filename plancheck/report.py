"""Human-readable outputs: a Markdown report and an overlay image of what was checked."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .agent import total_usage
from .synth import _font

PALETTE_OK = (46, 125, 50)
PALETTE_BAD = (198, 40, 40)


def overlay(state: dict, out_path: str | Path) -> Path:
    img = Image.open(state["image_path"]).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ppm = state["meta"].px_per_m
    failing = {r for res in state["results"] for r in res.failing_rooms}
    f = _font(max(12, int(ppm * 0.2)))
    for room in state["plan"].rooms:
        bad = room.id in failing
        c = PALETTE_BAD if bad else PALETTE_OK
        d.polygon(room.polygon, fill=c + (55,), outline=c + (255,), width=4)
        xs, ys = zip(*room.polygon)
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        info = next((g for g in state["geometry"]["rooms"] if g["id"] == room.id), None)
        if info:
            d.text((cx, cy + ppm * 0.35), f"{room.id} · {info['area_m2']} m²", fill=c + (255,), font=f, anchor="mm")
    exits = set(state["geometry"]["exits"])
    for door in state["plan"].doors:
        x, y = door.position
        r = ppm * 0.18
        col = (25, 118, 210, 255) if door.id in exits else (255, 143, 0, 255)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return _save(Image.alpha_composite(img, layer).convert("RGB"), out_path)


def _save(img: Image.Image, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return p


def markdown(state: dict) -> str:
    res = state["results"]
    n_fail = sum(not r.passed for r in res)
    u = total_usage(state)
    lines = [
        f"# PlanCheck report: {Path(state['image_path']).name}",
        "",
        f"**{len(res) - n_fail} of {len(res)} rules pass.** "
        f"{len(state['plan'].rooms)} rooms, {len(state['plan'].doors)} doors, "
        f"exits: {', '.join(state['geometry']['exits']) or 'none'}. "
        f"Extraction attempts: {state['attempts']}. Model calls: {u.calls}, "
        f"tokens in/out: {u.input_tokens}/{u.output_tokens}, model time: {u.seconds:.1f} s.",
        "",
        "| Rule | Requirement | Result | Details |",
        "|---|---|---|---|",
    ]
    for r in res:
        det = "<br>".join(r.failures) if r.failures else ""
        lines.append(f"| {r.rule_id} | {r.text} | {'✅ pass' if r.passed else '❌ fail'} | {det} |")
    lines += ["", "## Rooms", "", "| Id | Label | Type | Area (m²) | Min width (m) | Doors | Egress (m) |",
              "|---|---|---|---|---|---|---|"]
    for g in state["geometry"]["rooms"]:
        lines.append(f"| {g['id']} | {g['label']} | {g['type']} | {g['area_m2']} | {g['min_width_m']} | "
                     f"{', '.join(g['doors'])} | {g['egress_m'] if g['egress_m'] is not None else '—'} |")
    warn = [i for i in state.get("issues", []) if i["severity"] != "error"]
    err = [i for i in state.get("issues", []) if i["severity"] == "error"]
    if err or warn:
        lines += ["", "## Open geometry issues", ""] + [f"- **{i['severity']}**: {i['message']}" for i in err + warn]
    lines += ["", "## Agent trace", "", "```"] + state["log"] + ["```", ""]
    return "\n".join(lines)
