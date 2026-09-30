# Scrub report

Checks run over the working tree (and, after commit, `git log --all -p`) before any sharing.
Greps use GNU grep: `grep -rnI -E <pattern> --exclude-dir=.git .`

| # | Check | Pattern | Result |
|---|---|---|---|
| 1 | emails, phone numbers | `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}`; `\(?\b[0-9]{3}\)?[-. ][0-9]{3}[-. ][0-9]{4}\b` | 0 hits |
| 2 | tailnet names, private IPs | `\.tail[a-f0-9]+\.ts\.net\|\b100\.x.x.x\|192\.168\.\|\b10\.x.x.x\|172\.(16-31)\.` | 0 hits. Node IPs in capture/bench scripts were replaced with environment variables |
| 3 | machine names, usernames | local username, node and host names, `hostname` (case-insensitive) | only upstream vLLM code (`VLLM_ROCM_USE_SKINNY_GEMM`, `MOONCAKE_REQUESTER_LOCAL_HOSTNAME`). Node names in docs were rewritten as "head" / "worker node"; a host name was removed from a solver docstring |
| 4 | absolute paths | `/home/\|/Users/\|C:\\Users\|~/\|\$HOME` | only upstream vLLM defaults (`~/.cache/vllm`, `~/.config/vllm`) |
| 4b | container-internal paths | `/root/` | intended: `/root/.cache/...` is the in-container cache mount used by `launch/*/launch.sh`, `overlays/tp4-load-accel/{tr3_runtime,entrypoint}.py` and `experiments/tp4-profiles/PROFILE-PREFILL.md`. It is the container's path, not a host path |
| 4c | build paths inside binaries | `strings` on `third_party/e3/grouped_fragments.cubin` | the original cubin embedded local build paths in its debug line info. It was rebuilt from a neutral directory with the same compiler and flags; code, constant and info sections are byte-identical (17/17 sections compared). 0 hits now |
| 5 | secrets | `ghp_\|gho_\|github_pat_\|xox[abp]\|AKIA\|sk-…\|hf_…\|password\|passwd\|secret\|api[_-]?key\|token=` | only license text and code words (`row_token`, "secretly created" in upstream vLLM) |
| 6 | company branding, client names | company and client-name list | 0 hits |
| 7 | personal names | operator and team first names | 0 hits. Upstream authors are credited by their public names/handles (Matt Mastracci, MiaAI-Lab) |
| 8 | internal agent/model names | internal agent and assistant-model names | only upstream vLLM model-class names and an audio-codec name in upstream `envs.py`. A debug dir name was renamed in `third_party/e3/serving.py` |
| 12 | git history | same patterns over `git log --all -p` | see commit check below |
| 13 | tracked files | `git ls-files` count, sizes | see commit check below |

Also checked: `eval/kl-seqs.jsonl` is token ids only; decoded, it is 16 generic prompts
(short story, TCP vs UDP, curry recipe, …) and the model's continuations, with no private text.

Left out on purpose: the calibration corpus and its token ids (private text), campaign
hand-off notes, infra-specific staging/boot scripts, logs, traces, weights, and code whose
license was not verified. See README and THIRD_PARTY_NOTICES for what each piece references.
