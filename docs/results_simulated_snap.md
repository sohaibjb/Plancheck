# Ablation: wall snapping (simulated extractor, 30 synthetic plans)

The simulated extractor replays the ground truth with VLM-like errors (≈0.25 m coordinate jitter,
a missing room in 30% of first answers, an occasional wrong room type). It measures the
pipeline around the model, not any real model.

| Pipeline | Plans | Rule acc. | Plans fully right | Room F1 | Mean IoU | Type acc. | Area err. | Door recall | Attempts |
|---|---|---|---|---|---|---|---|---|---|
| extract → snap → validate/repair → check | 30 | 100.0% | 100.0% | 1.000 | 0.997 | 98.6% | 0.2% | 100.0% | 1.33 |
| extract → validate/repair → check (no snap) | 30 | 80.4% | 26.7% | 1.000 | 0.915 | 98.5% | 4.4% | 93.7% | 2.97 |

Snapping the model's coordinates onto the wall lines detected in the full-resolution image cuts the
area error from 4.4% to 0.2% and makes every rule verdict match the ground truth. Without it, most
plans also use up all repair attempts because the noisy polygons overlap.

Reproduce:

```bash
plancheck eval --models simulated --out results
plancheck eval --models simulated --no-snap --out results_nosnap
```
