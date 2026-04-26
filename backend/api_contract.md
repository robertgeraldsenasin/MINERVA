# MINERVA Unity Backend API Contract

This design assumes **Supabase Auth + Postgres + Row Level Security (RLS)**.

Why this stack:
- `cards` are structured JSON records, so **database queries** are a better fit than serving one giant static file.
- MINERVA needs **per-user exclusion** of already-used cards.
- Supabase exposes Postgres through REST/RPC while still applying RLS at the table level.

## Architecture

1. Offline MINERVA pipeline generates `story_cards.json`.
2. Admin imports those cards into `public.cards`.
3. Unity authenticates the player with Supabase Auth.
4. Unity calls RPC `get_next_cards(...)`.
5. The RPC excludes cards already marked `discarded` or `used` for that player.
6. After the player makes a decision, Unity calls `submit_card_result(...)`.

## Tables

### `public.cards`
Stores the canonical, game-ready card payload.

Important fields:
- `id`
- `deck_version`
- `day`
- `verdict`
- `truth_type`
- `is_misinformation`
- `targets`
- `linked_blue_truth_id`
- `text`
- `card_json`

### `public.user_card_state`
Tracks which cards a player has already seen / answered / used.

Important fields:
- `user_id`
- `card_id`
- `status` (`seen`, `answered`, `discarded`, `used`)
- `player_verdict`
- `shared`
- `confidence`
- `day`
- `metadata`

## Auth model

- Unity signs the player in using Supabase Auth.
- The JWT is placed in the `Authorization: Bearer <token>` header.
- RLS ensures each player can only read or write their own `user_card_state` rows.

## Endpoints

## 1) Get next cards
**POST** `/rest/v1/rpc/get_next_cards`

Request body:
```json
{
  "p_day": 4,
  "p_limit": 5,
  "p_candidate_ids": ["A", "B", "C"]
}
```

Response:
```json
[
  {
    "id": "card_fake_A_0001",
    "day": 4,
    "text": "BREAKING: ...",
    "targets": ["A"],
    "classification": {
      "truth_type": "red",
      "is_misinformation": true,
      "linked_blue_truth_id": "BT-01"
    }
  }
]
```

Behavior:
- Returns only `active = true` cards.
- Excludes cards already marked `discarded` or `used` for the current player.
- Filters by `day` when provided.
- Filters by candidate target overlap when `p_candidate_ids` is provided.

## 2) Submit player result
**POST** `/rest/v1/rpc/submit_card_result`

Request body:
```json
{
  "p_card_id": "card_fake_A_0001",
  "p_player_verdict": "fake",
  "p_shared": false,
  "p_confidence": 0.82,
  "p_day": 4,
  "p_metadata": {
    "time_seconds": 18.7,
    "screen": "VerDICT"
  }
}
```

Response:
```json
{
  "status": "ok",
  "card_id": "card_fake_A_0001",
  "shared": false,
  "saved_at": "2026-04-24T00:00:00Z"
}
```

Behavior:
- Upserts the player's state for the selected card.
- If `p_shared = true`, the card is marked `used`.
- If `p_shared = false`, the card is marked `answered`.
- This prevents the same player from receiving the same fully-used card later.

## 3) Admin import deck
Recommended options:
- service-role call to `import_cards(...)`
- or a protected server-side admin job that reads `story_cards.json` and inserts rows into `public.cards`

Do **not** expose deck-import permissions to the Unity client.

## Storage recommendation

Use the **database** for live card delivery, not raw file hosting.

When to use Supabase Storage:
- keep an archive copy of `story_cards.json`
- store reports / audit bundles / screenshots
- store downloadable patch zips

When **not** to use only Storage:
- if the client must exclude cards already used per player
- if the client needs day filtering, candidate filtering, or live analytics

## Versioning strategy

Recommended:
- `deck_version = v1`, `v2`, `v3`
- keep old cards in the database for reproducibility
- set `active = false` on retired deck versions
- keep import logs in `deck_versions`

## Example deployment flow

1. Generate cards offline:
```bash
python scripts/18_verdict_explain.py ...
python scripts/21_balance_unity_cards.py ...
python scripts/22_build_story_cards.py ...
```

2. Import to backend:
```sql
select public.import_cards(<story_cards_json>, 'v1');
```

3. Unity runtime:
- call `get_next_cards(...)`
- show one card in the feed
- open VerDICT analysis screen
- after player decision, call `submit_card_result(...)`

## Why this is stronger than serving one static JSON file

A static file can deliver cards, but it cannot reliably:
- stop repeat cards per authenticated player
- record player decisions
- support live balancing and analytics
- enforce secure, per-user visibility rules

That is why the recommended production design is:
- **Postgres table for cards**
- **RLS-protected per-user state table**
- **RPC for card delivery and result submission**
