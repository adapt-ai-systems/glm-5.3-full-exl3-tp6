#!/usr/bin/env python3
"""Per-workload per-position MTP acceptance + decode speed on the serving boot.
For each frozen lane-1 decode prompt (structured, code, prose; T0, 512 tok, 5 nonstream + 5 SSE), snapshot
/metrics around the whole name and report survival S_i = accepted_at_pos[i] / drafts (unconditional), tok/round,
conditional c_i, and median SSE generation tok/s -> round ms = 1000 * tok/round / tok/s.
MTP drafts are sequential, so S_1..S_j measured at k=K hold for any k=j <= K.
usage: accept.py OUTDIR [names]"""
import json, os, statistics, subprocess, sys, urllib.request, pathlib
U = os.environ.get('URL', 'http://localhost:8000'); H = os.environ['HARNESS_DIR']  # dir with the frozen prompt files, see eval/README.md
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
names = (sys.argv[2] if len(sys.argv) > 2 else 'structured,code,prose').split(',')

def snap():
    t = urllib.request.urlopen(U + '/metrics', timeout=30).read().decode()
    g = lambda k: sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(k))
    pos = [sum(float(l.split()[-1]) for l in t.splitlines()
               if l.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total') and f'position="{i}"' in l) for i in range(8)]
    return g('vllm:spec_decode_num_drafts_total'), pos

res = {}
for n in names:
    d0, p0 = snap()
    subprocess.run([sys.executable, f'{H}/decode.py', '--base', U + '/v1', '--model', 'glm-5.3', '--names', n,
                    '--reps', '5', '--out', str(out / f'decode-{n}.jsonl')], check=True, stdout=subprocess.DEVNULL)
    d1, p1 = snap()
    drafts = d1 - d0; S = [(b - a) / drafts for a, b in zip(p0, p1)] if drafts else []
    k = max([i + 1 for i, s in enumerate(S) if s > 0] or [0])
    S = S[:max(k, 1)]
    rows = [json.loads(l) for l in open(out / f'decode-{n}.jsonl')]
    sse = [r['sse_generation_tps'] for r in rows if r['mode'] == 'sse' and r.get('sse_generation_tps')]
    wall = [r['wall_tps'] for r in rows if r['mode'] == 'nonstream' and r.get('wall_tps')]
    tpr = 1 + sum(S); tps = statistics.median(sse) if sse else None
    c = [S[0]] + [S[i] / S[i - 1] if S[i - 1] else 0 for i in range(1, len(S))] if S else []
    res[n] = {'drafts': drafts, 'S': [round(x, 4) for x in S], 'c': [round(x, 4) for x in c],
              'tok_per_round': round(tpr, 3), 'sse_tps_med': tps, 'wall_tps_med': statistics.median(wall) if wall else None,
              'round_ms': round(1000 * tpr / tps, 2) if tps else None,
              'E_k': {j: round(1 + sum(S[:j]), 3) for j in range(1, len(S) + 1)}}
    print(n, json.dumps(res[n]), flush=True)
json.dump(res, open(out / 'accept.json', 'w'), indent=1)
