# Calibrating the marking detector

The lane-marking response has two parameters that decide everything downstream,
and they trade off against each other in a way worth documenting: too permissive
and the pipeline **invents** lane lines out of sensor grain; too strict and it
misses the paint that is genuinely there.

| parameter | meaning |
|---|---|
| `local_norm_gain` | how many times the local pavement response a pixel must reach to score 1.0 |
| `min_marking_contrast` | absolute floor on that local scale, so it cannot collapse on featureless asphalt |

## Why local normalisation is needed

A single global scale is set by whatever is brightest in the scene — typically
fresh parking-bay stripes on pale concrete. On the img93 tile that drove the
response on the dark, weathered arterial to nearly zero even where its lines
were visible. Dividing by the local response level over nearby pavement makes a
fixed threshold mean the same thing everywhere.

## Why an absolute floor is needed

Local normalisation alone has a failure mode that is worse than missing paint.
On a road with *no* markings at all, the local scale becomes the sensor noise
floor, and normalising by it lifts grain to marking level. The detector then
finds evenly spaced "lines" at plausible lane offsets, the lane-count
arbitration sees markings that agree with the corridor width, and the pipeline
emits a **confident, entirely fictional lane structure** with
`Source.FUSED` and confidence 0.85.

This was a real regression, caught by the synthetic test scene with the paint
deleted. An absolute floor on the local scale means "a marking must be brighter
than this road's own noise *and* brighter than a fixed contrast".

## The sweep

Three criteria, evaluated jointly:

1. **synthetic scene with paint** (`tests/conftest.py`, 4 lanes, sub-pixel paint
   rasterised by area coverage) — must recover all 4;
2. **same scene with paint deleted** — must recover *none* (`n_image is None`);
3. **three real 0.27 m tiles** (img 38, 10, 93) — maximise the share of
   major-road carriageways where markings are observed.

| gain | floor | synthetic w/ paint | synthetic bare | real major roads |
|---|---|---|---|---|
| 3.0 | 0.000 | 4 | **4** ✗ | 100.0 % |
| 3.0 | 0.012 | 4 | none | 94.3 % |
| **4.0** | **0.008** | **4** | **none** | **94.3 %** |
| 4.0 | 0.012 | 4 | none | 82.9 % |
| 5.0 | 0.008 | 4 | none | 80.0 % |
| 6.0 | 0.000 | 4 | 2 ✗ | 91.4 % |
| 6.0 | 0.012 | 4 | none | 54.3 % |

Rows with a non-`none` result in the "bare" column are disqualified regardless
of their real-data score: inventing lanes is not a trade-off, it is a defect.

**Chosen: `local_norm_gain=4.0`, `min_marking_contrast=0.008`** — the operating
point with the highest real-data recall among the rows that invent nothing.

## Caveat

This is calibrated on one city, one sensor and one season. Las Vegas is
high-contrast desert with mostly dry, pale ground; the same thresholds on wet
northern-European asphalt under overcast light would need re-checking. The
sweep above is cheap to re-run — it is three numbers from two synthetic scenes
and three tiles — and re-running it is the right move before trusting the
detector on a new region.
