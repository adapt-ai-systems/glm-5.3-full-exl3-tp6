# Working on this repo

README.md and RESULTS.md are what people compare their own runs against, so
a number changes only when it was measured. Read RESULTS.md's benchmark
definitions before adding a row.

## kindling.json

`kindling.json` at the repo root is the README's results table in a form a
program can read. The Kindling AI site reads it from `main`, so whatever is
merged there is what people see.

Keep it in step with the README:

- When a change moves a number in the README table (or the RESULTS.md "kept"
  rows it comes from), change `kindling.json` in the same commit or PR. Same
  numbers, same rounding.
- Publish the shipped configuration. Prose decode is the MTP k=4 number,
  because that is what the launchers run; other settings go in `notes`.
- Only numbers measured on the shipped stack. The concurrency table in
  RESULTS.md ran on the previous FP8 stack with `max-num-seqs 8`, so it stays
  out until it is re-run.
- One entry in `configs` per build (TP6, TP4). Leave out a number that wasn't
  measured. Don't write 0, and don't fill a gap with an estimate.
- Check it parses before you push: `python3 -m json.tool kindling.json`.

The fields:

| field | what |
|---|---|
| `title`, `variant`, `summary` | the card heading, the quant tag, one or two sentences |
| `model`, `hardware`, `stack` | the checkpoint, the boxes, a few short tags |
| `default_config` | the `id` of the config shown first |
| `configs[].id`, `label`, `nodes` | e.g. `tp4`, `TP=4`, `4` |
| `configs[].decode` | single-stream decode tok/s by workload: `code`, `prose`, `structured` |
| `configs[].prefill` | cold prefill, `[{ "context": tokens, "tps": tok/s }]` |
| `configs[].concurrency` | aggregate tok/s, `[{ "streams": n, "aggregate_tps": tok/s }]` |
| `configs[].max_context`, `kv_pool_tokens`, `max_concurrent` | integers: tokens, tokens, requests |
| `configs[].quality` | `[{ "name": ..., "value": ... }]`, shown as short text |
| `configs[].notes` | one line of caveats |

Context lengths are in tokens: 32k is `32768`, 128k is `131072`.
