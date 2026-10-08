# Plan: make meal sizes optional in the user config

Status: not started. Depends on the `salt-per-100g` branch landing first. It
adds `MealVariant.weight_g` and sends only salt per 100 g
(`nutrition_per_100g.salt_g`) to the LLM; the other nutrients stay
per-portion only. Without a configured size this plan needs per-100 g values
for every nutrient, so it adds the rest to that same block.

## Goal

`purchased_meals[].size` becomes optional so a new user only has to list the
meal types they bought. When a size is given, behaviour stays exactly as
today (exact portion values). When it is omitted, the app picks a size itself
and the LLM scores on per-100 g values only.

## Findings: how the ntfy API is structured

Checked against all 10 fixture days in `tests/fixtures/ntfy/` (offers 6 and
8, 170 dishes, 588 distinct products).

A day's response for one `diet_offer_id` is a flat list of `results` rows:
`diet_id`, `diet_variant_id`, `diet_variant_meal_type_id`,
`simple_product_id`, `configurable_product_id`, plus `includes` with
`diet_variant_meal_types` and `simple_products`.

- `diet_id` (3 per offer) is a diet line. The 3 dishes per meal type that
  the LLM scores are one dish from each diet line (snack: 2).
- `diet_variant_id` (7 per diet) is a calorie plan, roughly 1000 to 3000
  kcal. It decides the size of each meal, e.g. ~1000 kcal is S everywhere,
  ~2450 kcal is breakfast L / lunch XL / dinner XL / second breakfast L /
  tea L / snack S. Snack only appears in the two largest plans.
- `configurable_product_id` is the dish (already used as
  `provider_meal_id`, size-independent). `simple_product_id` is one size of
  it. The same product row repeats once per calorie plan that uses it, so a
  dish group contains duplicates (dedupe by `simple_product_id`).
- Size ladders per meal type: breakfast S–XL, lunch S–XXL, dinner S–XL,
  second breakfast S–L, tea S–L, snack S/L.

So configured sizes only encode the user's calorie plan. They decide which
`simple_product` is read, never which dishes are offered. Users may also
have per-meal sizes that match none of the 7 plans (the example config does),
so "configure a kcal plan instead" is not a general replacement.

### What differs between sizes of one dish

| Value | Same across sizes? | Evidence |
|---|---|---|
| `configurable_product_id` | yes | by definition |
| `name` | almost always | 1 of 170 dishes has a German name on one size (offer 8, 2026-07-16) |
| ingredient set | yes | identical in all 170; 3 "differences" are only reordered sub-ingredients |
| `composition` text | no, in 71 of 170 | only percentages and therefore ingredient order change ("frittata 60%" S vs "75%" XL) |
| nutrition (`protein`, `fat`, ..., `salt`, `calorific`) | no | totals per portion, scale with `weight` (frittata protein 15.5 g S vs 40.6 g XL) |
| nutrition per 100 g (derived from `weight`) | roughly | median spread between sizes ~5–7%, outliers up to 1.5–2× (e.g. dinners where bigger sizes get more of the protein component) |
| `protein_percent`, `carb_percent`, `fat_percent` | roughly | share of energy, the only ratio fields the API has; no fiber/sugar/salt equivalent |

## Design

### Config

```yaml
purchased_meals:
  - lunch                 # shorthand, no size
  - type: dinner          # same, long form
  - type: breakfast
    size: M               # optional: exact portion values
```

- `config/loader.py`: accept a plain string or a mapping with `type` and an
  optional `size`. Existing configs keep working unchanged.
- `PurchasedMeal.size: str | None = None`.

### ntfy normalizer (`providers/ntfy/normalizer.py`)

- Dedupe each dish group by `simple_product_id`.
- Size given: `_pick_size` as today (including `MenuUnavailableError` when
  that size isn't published yet).
- Size omitted: use the same size for every dish of that meal type so dishes
  are compared fairly. Rule: the sizes offered by all of the meal's dishes,
  ordered by weight, take the middle one (lower middle on an even count, so
  snack S/L picks S). If a dish lacks the chosen size (partially published),
  fall back to whatever that dish has; don't raise.
- Name: take the most common `name` across the dish's sizes (makes the
  German-name glitch harmless). Applies in both modes.
- Composition and nutrition come from the chosen product, so composition
  percentages match the numbers.
- The 3-dishes-per-meal check (2 for snack) is unchanged.

### Canonical menu / prompt

Each field keeps one meaning regardless of config:

- `nutrition_per_100g`: always present, all nutrients (extends the
  salt-only block from step 1). The LLM scores on it,
  so scores stay comparable across users and sizes.
- `nutrition` (per portion) and `weight_g`: only sent when the meal has a
  size configured. Without a size the reference-size totals would be
  misleading, so leave them out of `to_compact_dict()` for that meal (e.g. a
  flag on `CanonicalMeal`/`MealVariant` set by the normalizer).
- `prompts/app.md`: per-portion fields may be absent; the 4 g per-portion
  salt flag only applies when `nutrition` is present; the per-100 g salt flag
  always applies.

### mo-web (`delivery/mo_web.py`)

It currently sends per-portion `nutrition`. Without a size that is not the
user's portion. Option: send `nutrition` only when a size is configured and
add `nutrition_per_100g` always. That's additive, so `SCHEMA_VERSION` can
stay 1, but mo-web needs to read the new field to show anything for size-less
meals. Coordinate with the mo-web side.

### Example provider (`providers/example_provider.py`)

Mirror the same rule: size given, read that size; omitted, pick the middle
available size. Keep it minimal.

### Docs and fixtures

- `config/users.example.yaml`: show the shorthand plus one entry with `size`
  and explain what the size buys (exact portion values, per-portion salt
  warning).
- `README.md`: workflow step 2 ("purchased meal types and sizes"),
  Configuration section, "Adding a new provider" step 3 (size not published
  yet only applies when a size is configured).

### Tests

- `test_config_loader.py`: string shorthand, mapping without size, mapping
  with size, invalid entries.
- `test_normalizer_ntfy.py`: default size rule (same size across the meal's
  dishes, middle of the shared ladder, snack picks S), partial publication
  fallback, most-common-name rule, duplicate rows, per-portion fields absent
  without size. Keep the existing size-given tests.
- Add a canonical fixture for a size-less config next to
  `canonical_offer6_week_2026-06-29.json`.
- `test_normalizer_example_provider.py`, `test_ntfy_provider_adapter.py`,
  `test_mo_web.py`, `helpers.py`, `test_workflow.py`,
  `test_orchestrator_batch.py` as needed.

## Optional, not planned

If `purchased_meals` is omitted entirely, default to every meal type the
offer has. Left out: the API can't tell which meals a user bought (snack only
comes with the biggest plans).
