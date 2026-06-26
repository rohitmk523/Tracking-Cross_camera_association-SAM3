# Detection dataset card (OUR FOOTAGE ONLY)

- target classes: `['player', 'referee', 'ball']`
- val_mode: `explicit`
- split_map: `{'2399cfac': 'train', '2c490f1a': 'train', '922bff3b': 'train', '95d2ea95': 'train', '9eb51980': 'train', 'd0a9faef': 'train', 'd186e25e': 'train', 'd446fe8c': 'train', 'e6fba750': 'train', '29b51d57': 'valid', '74c4f686': 'valid', 'b68967fe': 'valid', 'c2a354fe': 'test'}`
- sources used: `['annotate_pool', '4cam_detection']`
- duplicate frames skipped (cross-source): 0
- frames dropped (empty after remap): 2
- frames UNRESOLVED to a known game (not admitted): 0
- frames off split_map (resolved but not selected): 0

## Per-split counts

| split | images | player | referee | ball | games |
|---|---|---|---|---|---|
| train | 2198 | 13203 | 1577 | 618 | e6fba750(1175), 2399cfac(128), 2c490f1a(128), 922bff3b(128), 9eb51980(128), d0a9faef(128), d186e25e(128), d446fe8c(128), 95d2ea95(127) |
| valid | 351 | 2123 | 408 | 206 | 29b51d57(128), 74c4f686(128), b68967fe(95) |
| test | 1200 | 9906 | 220 | 0 | c2a354fe(1200) |

> NOTE: `valid` is one or more WHOLE held-out games (cross-game validation), distinct from `test`; no temporal-tail carving. `test` (the held-out anchor game) stays untouched.
