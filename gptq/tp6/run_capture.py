#!/usr/bin/env python3
"""Drive a capture boot (overlay d2_fp8_capture.py, see gptq/README.md): calibration prefills -> 'eval' marker -> held-out prefills -> 'dump' marker
-> one request (triggers the dump in each rank's op body) -> wait for dumped-r<rank> on every node.
Markers are node-local, so every phase marker is touched on every node. Prefill only (max_tokens=1), one at a time.
Run from this dir (needs calib_ids.jsonl, seqs.jsonl). Copy the r<rank>/ dumps off the nodes afterwards for the solve."""
import json, os, subprocess, sys, time, urllib.request

URL = os.environ.get('URL', 'http://localhost:8000')
H = os.environ['HEAD_SSH']          # e.g. user@<head host>; workers are reached through the head
CAP = os.environ['CAPTURE_DIR']     # host dir bind-mounted rw at /d8cap in the capture boot, same path on every node
# rank -> worker fabric address (None = the head itself). RANK_HOSTS = "worker1 worker2 ..." in --node-rank order.
RANKS = {0: None, **{i + 1: h for i, h in enumerate(os.environ['RANK_HOSTS'].split())}}
FULL = '--deep-only' not in sys.argv


def post(ids):
    req = urllib.request.Request(URL + '/v1/completions', json.dumps(
        {'model': 'glm-5.3', 'prompt': ids, 'max_tokens': 1, 'temperature': 0}).encode(),
        {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        json.load(r)


def ssh(cmd, ip=None):
    if ip:
        cmd = f"ssh -o BatchMode=yes root@{ip} '{cmd}'"
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', H, cmd], check=True, capture_output=True, text=True).stdout


def all_nodes(cmd):
    return {r: ssh(cmd, ip) for r, ip in RANKS.items()}


def main():
    urllib.request.urlopen(URL + '/health', timeout=10)
    assert all(v.strip() == '' for v in all_nodes(f'ls {CAP}').values()), 'capture dir not empty: restage + reboot'
    calib = [json.loads(l)['ids'] for l in open('calib_ids.jsonl')]
    seqs = [json.loads(l) for l in open('seqs.jsonl')]
    t = time.time()
    for i, ids in enumerate(calib):
        post(ids)
        if i % 20 == 0:
            print(f'calib {i}/{len(calib)} {time.time() - t:.0f}s', flush=True)
    all_nodes(f'touch {CAP}/eval')
    for s in seqs:
        post(s['prompt_ids'] + s['cont_ids'])
    print(f'eval {len(seqs)} done {time.time() - t:.0f}s', flush=True)
    all_nodes(f'touch {CAP}/dump')
    post([154822, 154824, 9707])
    need = RANKS if FULL else {0: None}
    for _ in range(300):
        done = {r for r, ip in need.items() if f'dumped-r{r}' in ssh(f'ls {CAP}', ip)}
        if len(done) == len(need):
            break
        time.sleep(2)
    else:
        sys.exit(f'dump incomplete after 600s: have {sorted(done)}')
    for r, ip in need.items():
        print(f'r{r}:', ssh(f'ls {CAP}/r{r} | wc -l; du -sh {CAP}/r{r}', ip).split(), flush=True)
    print(f'capture done {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
