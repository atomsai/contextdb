# Trust policy

`recall_for_action` is empty until a fact is trusted. That is intentional.
This page is the matrix hosts asked for.

A fact is trusted only when `action_relevant` is true **and** one of:

1. `confirmed` is true, or
2. independent corroboration ≥ the class threshold, or
3. first-party shortcut: `user_stated` and `confidence` ≥ the policy
   threshold, and the slot class is not excluded.

Otherwise an action-relevant fact has `requires_confirmation=True`.

Non-action-relevant facts are recallable but never gate an action.

Injection-suspect and contested (unconfirmed) facts are never trusted.

Unknown slots (`entity`+`attribute` not in the vocabulary) cannot use the
first-party shortcut when `unknown_slots_untrusted` is true (default).

## Default policy

| source | confidence | corroborated | result |
|---|---|---|---|
| `user_stated` | ≥ 0.7 | 1 | trusted (first-party) |
| `user_stated` | < 0.7 | 1 | needs confirmation |
| `agent_inferred` | any | 1 | needs confirmation |
| `third_party` | any | 1 | needs confirmation |
| any | any | ≥ 2 | trusted |
| any | any | confirmed | trusted |
| omitted `source` | — | — | stored as `user_stated` + **warning** (HTTP/MCP: 400) |

Default thresholds: confidence `0.7`, corroboration `2`. Class overrides:
health/legal/identity `3`; money/booking/contact `2`.

## `TrustPolicy.hospital()` vs `.restaurant()`

| | default | restaurant | hospital |
|---|---|---|---|
| confidence threshold | 0.7 | 0.7 | 0.95 |
| default corroboration | 2 | 2 | 2 |
| health / legal corroboration | 3 | 3 | 3 |
| identity corroboration | 3 | 2 | 3 |
| money corroboration | 2 | 2 | 3 |
| first-party excluded | (none) | health, legal | health, legal, identity |

Hospital: a patient saying a dose is not enough. Restaurant: a booking
wish confirmed in-session can graduate; an allergy still needs a second
source.

```python
db = contextdb.init(trust_policy=TrustPolicy.hospital())
```
