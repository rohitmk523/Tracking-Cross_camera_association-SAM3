# 14 · Games & Clips (what to pull from S3 / Supabase)

Authoritative inventory of which games and clips to retrieve for training, tracking, testing,
and narration eval. Machine-readable copy: [`configs/games.json`](../configs/games.json).

## Sources
- **Video (S3)**: bucket `uball-videos-production`, court-a. Path per angle:
  ```
  court-a/<date>/<game_id>/<date>_<game_id>_<ANGLE>.mp4     ANGLE ∈ {FL, FR, NL, NR}
  ```
  e.g. `court-a/2026-03-18/e6fba750-596a-441d-b216/2026-03-18_e6fba750-596a-441d-b216_FL.mp4`
- **Metadata + ground truth (Supabase)**: project `mhbrsftxvxxtfgbajrlc`
  - `games` — date, `team1/2_display_name`, **`team1/2_color`** (jersey colors for team
    classification), **`roster_team1/2`** (jsonb `[{player_id, name, jersey_number}]` — the
    per-game jersey→name map; replaces hand-built roster JSONs), scores.
  - `plays` (41k rows) — **descriptive annotations**: `classification` (FG_MAKE/FG_MISS/
    3PT_MAKE/TOV/REB), `player_a/b`, `events` (jsonb), `note` (human/Gemini summary),
    `start/end_timestamp`, `angle`. **This is gold for evaluating world-model events + VLM
    narration** (compare our descriptive output to these).
  - `video_metadata` — `s3_key`, `duration`, and **`sync_offset_seconds`** (per-angle sync
    offset; `per_angle_time = base_time + sync_offset` when the game's
    `timestamps_in_base_coords = true`). Use this for cross-camera sync.

## Clip-window convention (quick tests)
The demo used a 60 s window: **`t0 = frame 1400`, `dur = 60 s`, `fps = 30`**, decode filter
`field=top,scale=1920:1080,fps=30`. For targeted clips, slice around a `plays` event
timestamp (e.g. a specific shot/turnover) for exactly the moment under test.

## Working set (25 games — all 4 angles complete, annotated)
Splits are the established whole-game splits from the shot-detection manifest; we **adopt the
same whole-game discipline** (no game spans two splits; test opened sparingly). `plays` =
annotated plays available as ground truth.

|gid8|date|split|shots|plays|team1 (color)|team2 (color)|roster|
|---|---|---|---|---|---|---|---|
|`13e1ffad`|2026-01-31|train|122|206|A Day in Miami (Yellow)|Premier Mtg (Black)|7/8|
|`e6fba750`|2026-03-18|train|142|219|Blessed & Highly Favored (Blue)|Miracle Leaf (Green)|6/8|
|`2c490f1a`|2026-04-16|train|153|218|— (Gray)|— (Orange)|10/10|
|`922bff3b`|2026-04-16|train|137|229|— (Black)|— (Grey)|9/10|
|`d0a9faef`|2026-04-17|train|142|218|— (Blue)|— (Black)|9/10|
|`9eb51980`|2026-04-17|train|160|245|— (Green)|— (Red)|7/9|
|`d186e25e`|2026-04-18|train|158|194|— (Red)|— (Yellow)|7/10|
|`2399cfac`|2026-04-28|train|201|244|— (Black)|— (Blue)|12/7|
|`8dcb1330`|2026-04-28|train|167|208|— (Black)|— (Yellow)|10/7|
|`95d2ea95`|2026-04-29|train|179|223|— (White)|— (Black)|7/9|
|`cd045da8`|2026-04-29|train|162|208|— (White)|— (Blue)|8/8|
|`0fa23810`|2026-05-15|train|144|188|— (White)|— (Black)|9/7|
|`d446fe8c`|2026-05-15|train|166|209|— (Grey)|— (White)|8/9|
|`f66eb3b2`|2026-05-15|train|161|203|— (White)|— (Black)|9/9|
|`29b51d57`|2026-04-16|**val**|164|252|— (White)|— (Blue)|9/7|
|`74c4f686`|2026-04-17|**val**|147|221|— (Black)|— (Red)|7/7|
|`b68967fe`|2026-04-28|**val**|171|221|— (Grey)|— (Yellow)|12/6|
|`c2a354fe`|2026-03-19|**test**|189|315|Uptown 66ers (Black)|Los Mananeros (White)|10/9|
|`ee8745f1`|2026-04-16|**test**|160|202|— (Red)|— (White)|7/9|
|`6d601c99`|2026-04-18|**test**|136|192|305 Turnovers (Blue)|? (Gray)|11/8|
|`f3e7b25a`|2026-05-16|fresh|0|190|— (Black)|— (Blue)|8/10|
|`b3c1f62c`|2026-05-16|fresh|0|195|— (Blue)|— (White)|8/10|
|`77715f25`|2026-05-19|fresh|0|240|— (Red)|— (Blue)|8/9|
|`cc5deb39`|2026-05-19|fresh|0|204|— (Grey)|— (Yellow)|11/8|
|`cc1710c4`|2026-05-19|fresh|0|214|— (Navy)|— (Grey)|9/6|

> "fresh" = all-4-angle + roster/plays present but not in the shot-detection split; good
> **held-out generalization** candidates for tracking.

## Key reference games (used in DEMO_UBALL)
- **`e6fba750`** (Blue vs Green) — the demo game; distinctive jersey colors.
- **`c2a354fe`** (Black vs White) — v1-validated anchor; **keep held-out (test)** so tracking
  metrics stay comparable. Black/white kit = the hard same-brightness team case.
- **`d446fe8c`** (Grey vs White) — prior "newer-game" generalization eval target.
- **`9eb51980`, `2399cfac`** — prior multi-game SAM3 runs (cached dumps existed).

## Color diversity (good for general team classification)
Across the set: Blue/Green, Black/White, Gray/Orange, Black/Grey, Blue/Black, Green/Red,
Red/Yellow, Grey/Yellow, Navy/Grey, White/Blue… — wide kit variety, which is exactly what a
general team classifier needs (the demo's black-vs-white was the worst case).

## All complete 4-angle games on S3 (57 total — superset for expansion)
Beyond the 25 annotated working games, **32 more** have all 4 angles (no shot annotation yet —
annotate as needed). Full list in [`configs/games.json`](../configs/games.json) →
`all_complete_4angle_games`. Dates span **2026-01-20 … 2026-05-22** on court-a.

## ⚠️ Security note (Supabase)
The DB advisory flags **RLS disabled** on `public.score_events` and several `_backup_*` tables
(readable/writable by anyone with the anon key). Not ours to fix unilaterally, but flag to the
data owner: enable RLS + policies on those tables.

## How to pull (example)
```bash
SRC=s3://uball-videos-production/court-a/2026-03-18/e6fba750-596a-441d-b216
for A in FL FR NL NR; do
  aws s3 cp "$SRC/2026-03-18_e6fba750-596a-441d-b216_${A}.mp4" data/games/e6fba750/${A}.mp4
done
```
(Or stream a window via `aws s3 presign … | ffmpeg -ss <t0> -t <dur> -i URL …` to avoid full
multi-GB downloads — see [03](03_cameras_and_calibration.md).)
