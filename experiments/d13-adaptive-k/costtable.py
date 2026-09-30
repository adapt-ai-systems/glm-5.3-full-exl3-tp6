#!/usr/bin/env python3
"""Measure the D13 step-cost table on the live D13 boot: for each forced k, single-stream T0 decode of
2 prompts x REPS, round ms = sum(SSE decode seconds) / (drafts delta from /metrics).
Prints "cost 2:ms,3:ms,..." (verified tokens = k+1) and writes OUT json.  usage: costtable.py OUT.json [ks] [reps]"""
import json, os, subprocess, sys, time, urllib.request
U = os.environ.get('URL', 'http://localhost:8000')
CTL = os.environ['CONTROL_FILE']      # head-host path of VLLM_ADAPTIVE_K_CONTROL (bind-mounted into the container)
HEAD_SSH = os.environ.get('HEAD_SSH')  # set when the head is remote
out = sys.argv[1]; ks = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '1,2,3,4').split(',')]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
P = ['Write a 500-word story about a clockmaker who repairs a broken town clock. Plain prose.',
     'Write a Python module implementing an LRU cache class with get/put, a TTL option and full docstrings, then unit tests.']

def ctl(s):
    if HEAD_SSH:
        subprocess.run(['ssh', HEAD_SSH, f'printf "%s\\n" "{s}" > {CTL}'], check=True)
    else:
        open(CTL, 'w').write(s + '\n')

def drafts():
    t = urllib.request.urlopen(U + '/metrics', timeout=30).read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith('vllm:spec_decode_num_drafts_total'))

def run(prompt):
    p = {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 500, 'temperature': 0,
         'stream': True, 'stream_options': {'include_usage': True}}
    r = urllib.request.Request(U + '/v1/chat/completions', json.dumps(p).encode(), {'Content-Type': 'application/json'})
    first = last = None; n = 0
    for line in urllib.request.urlopen(r, timeout=900):
        line = line.decode().strip()
        if not line.startswith('data:') or line.endswith('[DONE]'):
            continue
        d = json.loads(line[5:])
        if d.get('usage'):
            n = d['usage']['completion_tokens']
        if d.get('choices'):
            now = time.time(); first = first or now; last = now
    return last - first, n

res = {}
for k in ks:
    ctl(f'force {k}'); time.sleep(2); run('Say hi.')  # control applies from the next schedule
    d0 = drafts(); secs = toks = 0
    for _ in range(reps):
        for p in P:
            s, n = run(p); secs += s; toks += n
    dr = drafts() - d0
    res[k] = {'round_ms': round(1000 * secs / dr, 2), 'tok_per_round': round(toks / dr, 3), 'tok_s': round(toks / secs, 2), 'drafts': dr}
    print(k, json.dumps(res[k]), flush=True)
json.dump(res, open(out, 'w'), indent=1)
print('cost ' + ','.join(f'{k + 1}:{res[k]["round_ms"]}' for k in sorted(res)))
