# Detection dataset card (OUR FOOTAGE ONLY)

- target classes: `['player', 'referee', 'ball']`
- val_mode: `temporal_holdout`
- split_map: `{'e6fba750': 'train', 'c2a354fe': 'test'}`
- sources used: `['4cam_detection']`
- duplicate frames skipped (cross-source): 0
- frames dropped (empty after remap): 0
- frames UNRESOLVED to a known game (not admitted): 0
- frames off split_map (resolved but not selected): 0

## Per-split counts

| split | images | player | referee | ball | games |
|---|---|---|---|---|---|
| train | 1058 | 6389 | 428 | 0 | e6fba750(1058) |
| valid | 117 | 764 | 47 | 0 | e6fba750(117) |
| test | 1200 | 9906 | 220 | 0 | c2a354fe(1200) |

> NOTE: `valid` is an in-domain temporal tail of the TRAIN game (for early-stopping); `test` (the held-out game) is untouched. True cross-court val awaits a 2nd venue / 3rd game (docs/14).
