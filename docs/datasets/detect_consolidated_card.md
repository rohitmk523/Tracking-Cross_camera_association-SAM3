# Detection dataset card (OUR FOOTAGE ONLY)

- target classes: `['player', 'referee', 'ball']`
- val_mode: `explicit`
- split_map: `{'13e1ffad': 'train', 'e6fba750': 'train', '2c490f1a': 'train', '922bff3b': 'train', 'd0a9faef': 'train', '9eb51980': 'train', 'd186e25e': 'train', '2399cfac': 'train', '8dcb1330': 'train', '95d2ea95': 'train', 'cd045da8': 'train', '0fa23810': 'train', 'd446fe8c': 'train', 'f66eb3b2': 'train', '29b51d57': 'valid', '74c4f686': 'valid', 'b68967fe': 'valid', 'c2a354fe': 'test', 'ee8745f1': 'test', '6d601c99': 'test', '77715f25': 'train', 'b3c1f62c': 'train', 'cc1710c4': 'train', 'cc5deb39': 'valid', 'f3e7b25a': 'test'}`
- sources used: `['annotate_pool_events', 'annotate_pool', '4cam_detection']`
- duplicate frames skipped (cross-source): 0
- frames dropped (empty after remap): 2
- frames UNRESOLVED to a known game (not admitted): 0
- frames off split_map (resolved but not selected): 0

## Per-split counts

| split | images | player | referee | ball | games |
|---|---|---|---|---|---|
| train | 2553 | 15780 | 1953 | 885 | e6fba750(1175), 2399cfac(176), 922bff3b(176), 2c490f1a(164), 95d2ea95(158), 9eb51980(128), d0a9faef(128), d186e25e(128), d446fe8c(128), 0fa23810(48), 13e1ffad(48), 77715f25(48), 8dcb1330(48) |
| valid | 447 | 2869 | 516 | 279 | 29b51d57(176), 74c4f686(176), b68967fe(95) |
| test | 1248 | 10282 | 276 | 41 | c2a354fe(1200), 6d601c99(48) |

> NOTE: `valid` is one or more WHOLE held-out games (cross-game validation), distinct from `test`; no temporal-tail carving. `test` (the held-out anchor game) stays untouched.

## Warnings
- annotate_pool_events: approved_only -> 499 reviewed frames admitted
