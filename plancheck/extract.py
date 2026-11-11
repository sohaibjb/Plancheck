"""Reading the drawing.

`VLMExtractor` asks a vision-language model (any LangChain chat model that accepts images) for
a `FloorPlan` via structured output. `SimulatedExtractor` replays the ground truth with noise and
mistakes, so the whole pipeline can be run and tested without an API key. It is a test double,
not a model.
"""

from __future__ import annotations

import base64
import io
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

from .schema import ROOM_TYPES, Door, FloorPlan, FloorPlanOut, PlanMeta, Room

MAX_SIDE = 1568  # most VLMs downscale above this; we resize ourselves so coordinates stay consistent


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    calls: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens,
                     self.seconds + other.seconds, self.calls + other.calls)


class Extractor(Protocol):
    name: str

    def extract(self, image_path: str, meta: PlanMeta, feedback: str | None = None,
                previous: FloorPlan | None = None) -> tuple[FloorPlan, Usage]: ...


SYSTEM = """You read architectural floor plans and return their rooms and doors as structured data.
Rules:
- Coordinates are pixels in the image you are given: origin top-left, x to the right, y down.
- Draw each room polygon along the CENTRE LINE of its surrounding walls, so neighbouring rooms share edges exactly.
  Follow corners in order; rectangular rooms have 4 points.
- Include every enclosed space: rooms, halls, corridors, bathrooms, storage.
- Room type must be one of: {types}. Halls and passages are 'corridor'. Use the label written on the drawing.
- A door is a gap in a wall, usually with a swing arc. Give the centre of the gap on the wall line.
  Include doors to the outside.
- Do not invent rooms or doors that are not drawn."""


def _image_payload(path: str, max_side: int) -> tuple[str, float, tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    W, H = img.size
    s = min(1.0, max_side / max(W, H))
    if s < 1.0:
        img = img.resize((round(W * s), round(H * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), s, img.size


def _scale_plan(plan: FloorPlan, k: float) -> FloorPlan:
    return FloorPlan(
        rooms=[Room(id=r.id, label=r.label, type=r.type, polygon=[(x * k, y * k) for x, y in r.polygon])
               for r in plan.rooms],
        doors=[Door(id=d.id, position=(d.position[0] * k, d.position[1] * k),
                    width_px=None if d.width_px is None else d.width_px * k) for d in plan.doors],
    )


@dataclass
class VLMExtractor:
    """Any LangChain chat model with vision, e.g. 'anthropic:claude-sonnet-5-5' or 'openai:gpt-5'."""

    model: str
    max_side: int = MAX_SIDE
    temperature: float | None = None  # some reasoning models reject a temperature; leave unset by default
    name: str = field(init=False)

    def __post_init__(self) -> None:
        from langchain.chat_models import init_chat_model

        self.name = self.model
        kw = {} if self.temperature is None else {"temperature": self.temperature}
        llm = init_chat_model(self.model, **kw)
        self._llm = llm.with_structured_output(FloorPlanOut, method="json_schema", include_raw=True)

    def extract(self, image_path, meta, feedback=None, previous=None):
        from langchain_core.messages import HumanMessage, SystemMessage

        b64, s, (w, h) = _image_payload(image_path, self.max_side)
        text = f"The image is {w} x {h} pixels. Extract all rooms and doors."
        if feedback and previous is not None:
            prev = FloorPlanOut.from_plan(_scale_plan(previous, s)).model_dump_json()
            text += ("\n\nYour previous answer failed these geometry checks:\n" + feedback +
                     "\nFix them. Previous answer:\n" + prev)
        msgs = [
            SystemMessage(SYSTEM.format(types=", ".join(ROOM_TYPES))),
            HumanMessage(content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]),
        ]
        t0 = time.perf_counter()
        out = self._llm.invoke(msgs)
        dt = time.perf_counter() - t0
        if out.get("parsing_error") or out.get("parsed") is None:
            raise ValueError(f"model output did not match the schema: {out.get('parsing_error')}")
        um = getattr(out["raw"], "usage_metadata", None) or {}
        usage = Usage(um.get("input_tokens", 0), um.get("output_tokens", 0), dt, 1)
        return _scale_plan(out["parsed"].to_plan(), 1 / s), usage


@dataclass
class SimulatedExtractor:
    """Replays the ground-truth JSON next to the image with VLM-like errors.

    - every coordinate is jittered by ~`noise_m` metres;
    - with probability `p_drop_room`, the first attempt forgets a room (the repair loop should catch the gap);
    - with probability `p_wrong_type`, one bedroom is labelled as 'other'.
    """

    noise_m: float = 0.25
    p_drop_room: float = 0.3
    p_wrong_type: float = 0.1
    seed: int = 0
    name: str = "simulated"

    def extract(self, image_path, meta, feedback=None, previous=None):
        truth = json.loads(Path(image_path).with_suffix(".json").read_text())
        plan = FloorPlan.model_validate(truth["plan"])
        rng = random.Random(f"{self.seed}:{Path(image_path).stem}:{bool(feedback)}")
        sd = self.noise_m * meta.px_per_m
        j = lambda v: v + rng.gauss(0, sd / 2)  # noqa: E731
        rooms = [Room(id=r.id, label=r.label, type=r.type, polygon=[(j(x), j(y)) for x, y in r.polygon])
                 for r in plan.rooms]
        if not feedback and rng.random() < self.p_drop_room and len(rooms) > 3:
            rooms.pop(rng.randrange(2, len(rooms)))
        if rng.random() < self.p_wrong_type:
            beds = [r for r in rooms if r.type == "bedroom"]
            if beds:
                beds[0].type = "other"
        doors = [Door(id=d.id, position=(j(d.position[0]), j(d.position[1])), width_px=d.width_px)
                 for d in plan.doors]
        return FloorPlan(rooms=rooms, doors=doors), Usage(calls=1)


def make_extractor(spec: str, **kw) -> Extractor:
    return SimulatedExtractor(**kw) if spec == "simulated" else VLMExtractor(spec, **kw)
