# Long-term Re-ID layer — design (operator spec, Jul 9)

Sits **on top of** the geometric tracking foundation (SAM3 single-object masklets →
4-camera court fusion). Handles the cases geometry alone cannot: a player **leaves the
court, subs out, or vanishes into a dead-ball scrum**, then returns.

## Two-tier identity

1. **Confirmed identity** — `(team_colour, jersey_number)`. Assigned when the jersey
   number is read with confidence. This is the real player (e.g. `B#11`).
2. **Temporary identity** — `temp_<n>`. Assigned when a track appears (geometry + team
   colour) but the **number is not yet readable**. Fully tracked, just not yet named.

## Re-tag flow (when a player re-appears)

```
player re-appears in a camera
      │
      ├─ jersey NUMBER readable?
      │        ├─ YES → re-tag to the matching confirmed (team, number)
      │        │         (offline: relabel the WHOLE temp segment, backward + forward)
      │        └─ NO  → keep as temp_<n>, tracked by geometry + team colour + ReID
      │
      └─ later, number becomes readable on the temp track
               → MERGE temp_<n> into its confirmed identity, retroactively
```

## Why this works for us specifically (batch / offline)

- **Retroactive reconnection**: because we process the whole game at once, the moment a
  temp track's number is ever read, we relabel its *entire* history — the temp segment
  is stitched to the confirmed identity in both time directions. No live tracker can do
  this; we can.
- **ReID bridges the gap even without a number**: the appearance embedding (torso colour
  + body) links a temp track to a recently-lost confirmed track by similarity, so a
  scrum exit often re-attaches before any number is read; the number, when it comes,
  only *confirms* and raises confidence.
- **Team colour is the coarse gate**: a temp track can only merge into a confirmed
  identity of the same team — halves the candidate set instantly, and is readable even
  when the number is not.

## Priority (operator directive)

Tracking-first. The geometric foundation (SAM3 single-object × 4 → court fusion) must
work **with or without** number OCR. Jersey number is the **confidence / re-id layer**,
added after the foundation is proven — it upgrades temp→confirmed and resolves
ambiguous re-entries, it is not a dependency for basic tracking.

## Build order

1. ✅ SAM3 single-object masklets per player per camera (AWS).
2. ⏳ Geometric 4-camera court fusion + leakage-free scoring vs operator GT.
3. ▢ Temp-id assignment for unnamed tracks + ReID appearance embedding per track.
4. ▢ Jersey-anchor retroactive merge (temp → confirmed), whole-game.
5. ▢ Long-clip (full-game) validation of leave/return over many events.
