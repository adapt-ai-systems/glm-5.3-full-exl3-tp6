#!/usr/bin/env python3
"""D8 quality check: next-token distribution drift of a weight format vs a reference,
scored on FIXED token sequences through the server's prompt_logprobs.

  gen     --url REF  --out seqs.jsonl       greedy continuations from the reference boot
  score   --url X    --seqs seqs.jsonl --out X.jsonl   top-K prompt logprobs per position
  compare REF.jsonl X.jsonl                 KL(ref||x) (top-K + lumped tail), top-1
                                            agreement, mean NLL delta of the ref tokens

Score every arm with GLM_D2_FP8_DEQUANT_M=1000000 so ALL rows (the scored prefill too)
take the small-M path whose weights are under test (4-bit in D8, Marlin FP8 in D2c);
otherwise prompt_logprobs would run through the FP8 big-M path and hide the 4-bit error.
Only /v1/completions, /tokenize and /detokenize are used (vLLM OpenAI server).
"""
import argparse
import json
import math
import urllib.request

PROMPTS = [
    'Write a vivid short story about a lighthouse keeper who finds a letter in a bottle.',
    'Write an essay on why small towns change slowly.',
    'Describe a thunderstorm rolling over a wheat farm at night.',
    'Explain how a bill becomes law in the United States, step by step.',
    'Write a Python function that merges overlapping intervals, with a docstring and tests.',
    'Prove that the square root of 2 is irrational.',
    'Summarize the causes of the French Revolution in plain language.',
    'Write a friendly email declining a meeting invitation and proposing two alternatives.',
    'Explain the difference between TCP and UDP to a junior engineer.',
    'Write a sonnet about autumn in a city.',
    'Give a recipe for a weeknight vegetable curry with exact quantities.',
    'Explain gradient descent and why the learning rate matters.',
    'Write a SQL query that finds the top 3 customers by revenue per region, and explain it.',
    'Describe the water cycle for a ten-year-old.',
    'Write a product description for a waterproof hiking backpack.',
    'Compare the economic policies of Keynes and Hayek.',
]


def post(url, path, payload):
    r = urllib.request.Request(url + path, json.dumps(payload).encode(),
                               {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=1800))


def chat_prefix(url, text):
    # the served chat template, thinking off (same as pbench effort=nothink)
    t = post(url, '/tokenize', {'model': 'glm-5.3', 'add_generation_prompt': True,
                                'messages': [{'role': 'user', 'content': text}],
                                'chat_template_kwargs': {'enable_thinking': False}})
    return t['tokens']


def gen(a):
    with open(a.out, 'w') as f:
        for i, p in enumerate(PROMPTS[:a.n]):
            ids = chat_prefix(a.url, p)
            r = post(a.url, '/v1/completions', {'model': 'glm-5.3', 'prompt': ids,
                                                'max_tokens': a.tokens, 'temperature': 0,
                                                'return_tokens_as_token_ids': True,
                                                'logprobs': 1})
            toks = [int(t.split(':')[1]) for t in r['choices'][0]['logprobs']['tokens']]
            f.write(json.dumps({'i': i, 'prompt_ids': ids, 'cont_ids': toks}) + '\n')
            print(i, len(ids), len(toks), flush=True)


def score(a):
    with open(a.seqs) as fi, open(a.out, 'w') as fo:
        for line in fi:
            s = json.loads(line)
            ids = s['prompt_ids'] + s['cont_ids']
            r = post(a.url, '/v1/completions', {'model': 'glm-5.3', 'prompt': ids, 'max_tokens': 1,
                                                'temperature': 0, 'prompt_logprobs': a.k})
            pl = r['choices'][0].get('prompt_logprobs') or r.get('prompt_logprobs')
            n0 = len(s['prompt_ids'])
            pos = []
            for j in range(n0, len(ids)):  # distribution that predicts token ids[j]
                d = pl[j]
                pos.append({'tok': ids[j], 'top': {int(t): v['logprob'] for t, v in d.items()}})
            fo.write(json.dumps({'i': s['i'], 'pos': pos}) + '\n')
            print(s['i'], len(pos), flush=True)


def kl_topk(p, q, floor=-30.0):
    """KL(p||q) over the union of both top-K sets; the rest of each distribution is lumped
    into one tail bucket. Tokens missing from q's top-K get q's tail mass spread evenly
    over the missing ones (an upper-bound-ish estimate; exact when both top-Ks match)."""
    keys = set(p) | set(q)
    pt = 1 - sum(math.exp(v) for v in p.values())
    qt = 1 - sum(math.exp(v) for v in q.values())
    miss_p = [t for t in keys if t not in p]
    miss_q = [t for t in keys if t not in q]
    kl = 0.0
    for t in keys:
        lp = p.get(t)
        lq = q.get(t)
        pv = math.exp(lp) if lp is not None else max(pt, 0) / max(len(miss_p), 1)
        qv = math.exp(lq) if lq is not None else max(qt, 0) / max(len(miss_q), 1)
        if pv > 0:
            kl += pv * (math.log(pv) - math.log(max(qv, math.exp(floor))))
    pr = max(pt - sum(max(pt, 0) / max(len(miss_p), 1) for _ in miss_p), 0)
    qr = max(qt - sum(max(qt, 0) / max(len(miss_q), 1) for _ in miss_q), 0)
    if pr > 1e-12:
        kl += pr * (math.log(pr) - math.log(max(qr, math.exp(floor))))
    return max(kl, 0.0)


def compare(a):
    ref = {json.loads(l)['i']: json.loads(l) for l in open(a.ref)}
    x = {json.loads(l)['i']: json.loads(l) for l in open(a.x)}
    kls, agree, dnll, n = [], 0, 0.0, 0
    for i in sorted(ref):
        for pr, px in zip(ref[i]['pos'], x[i]['pos']):
            assert pr['tok'] == px['tok']
            p, q = pr['top'], px['top']
            kls.append(kl_topk(p, q))
            agree += max(p, key=p.get) == max(q, key=q.get)
            t = pr['tok']
            if t in p and t in q:
                dnll += p[t] - q[t]
                n += 1
    kls.sort()
    m = len(kls)
    print(json.dumps({'positions': m, 'mean_kl': round(sum(kls) / m, 5),
                      'p50_kl': round(kls[m // 2], 5), 'p99_kl': round(kls[int(m * 0.99)], 4),
                      'top1_agree': round(agree / m, 4),
                      'mean_nll_delta_ref_tokens': round(dnll / max(n, 1), 5)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    g = sp.add_parser('gen'); g.add_argument('--url', required=True); g.add_argument('--out', required=True)
    g.add_argument('--n', type=int, default=16); g.add_argument('--tokens', type=int, default=256)
    s = sp.add_parser('score'); s.add_argument('--url', required=True); s.add_argument('--seqs', required=True)
    s.add_argument('--out', required=True); s.add_argument('--k', type=int, default=20)
    c = sp.add_parser('compare'); c.add_argument('ref'); c.add_argument('x')
    a = ap.parse_args()
    {'gen': gen, 'score': score, 'compare': compare}[a.cmd](a)
