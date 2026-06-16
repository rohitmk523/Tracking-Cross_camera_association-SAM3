# Detection dataset card (OUR FOOTAGE ONLY)

- target classes: `['player', 'referee', 'ball']`
- val_mode: `held_out_game`
- split_map: `{'e6fba750': 'train', 'c2a354fe': 'test'}`
- sources used: `['4cam_detection']`
- duplicate frames skipped (cross-source): 0
- frames dropped (empty after remap): 0

## Per-split counts

| split | images | player | referee | ball | games |
|---|---|---|---|---|---|
| train | 1175 | 7153 | 475 | 0 | e6fba750(1175) |
| valid | 1200 | 9906 | 220 | 0 | c2a354fe(1200) |
| test | 1200 | 9906 | 220 | 0 | c2a354fe(1200) |

> NOTE: `valid` == `test` (single held-out game). Early-stopping uses the held-out game as a cross-game signal; reported test == val. Annotate a 3rd game (from the *fresh* set) to make val != test.
