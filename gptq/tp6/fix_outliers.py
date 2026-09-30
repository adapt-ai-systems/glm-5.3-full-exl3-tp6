"""Re-solve GPTQ modules whose dequant-vs-BF16 weight rel err exceeds MAXW (0.2) with stronger damping
(percdamp 0.03 -> 0.1 -> 0.3 -> 1.0, first that brings it <= MAXW), rewrite their rank/layer files in place
(originals kept under out_orig/). Usage: fix_outliers.py <out dir> <dense dir> <cap dir> [group=64]"""
import json, shutil, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import solve_gptq as S

out, dense, cap = map(Path, sys.argv[1:4])
MAXW, G = 0.2, int(sys.argv[4]) if len(sys.argv) > 4 else 64
torch.set_num_threads(14)
fixed = []
for L in range(78):
    with safe_open(dense / f'layer-{L:03d}.safetensors', 'pt') as f:
        for r in range(6):
            W = S.rank_weights(f, L, r)
            path = out / f'r{r}' / f'layer-{L:03d}.safetensors'
            with safe_open(path, 'pt') as o:
                meta, tens = o.metadata(), {k: o.get_tensor(k) for k in o.keys()}
            changed = False
            for p, w in W.items():
                dq = lambda C, Sc: S.dequant(C, Sc, G).to(torch.bfloat16).float()
                relw = lambda q: ((q - w).norm() / w.norm()).item()
                e0 = relw(dq(S.unpack(tens[p + '.q4']), tens[p + '.s4'].float()))
                if e0 <= MAXW:
                    continue
                Hs, n, hr = S.load_cap(cap, p, [L % 6, 0] if S.replicated(p) else [r])
                for d in (0.03, 0.1, 0.3, 1.0):
                    C, Sc = S.gptq_codes(w, Hs, G, True, percdamp=d)
                    e = relw(dq(C, Sc))
                    if e <= MAXW:
                        break
                pg, pr = S.proxy_err(w, S.dequant(C, Sc, G), Hs), S.proxy_err(w, S.WE.int4(w, G, clip=True), Hs)
                tens[p + '.q4'], tens[p + '.s4'] = S.pack(C), Sc.to(torch.bfloat16)
                changed = True
                rec = dict(layer=L, rank=r, module=p.split('.', 3)[3], relw_before=round(e0, 4), percdamp=d, relw_after=round(e, 4),
                           proxy_gptq=round(pg, 5), proxy_rtn=round(pr, 5))
                fixed.append(rec); print(json.dumps(rec), flush=True)
            if changed:
                bk = out.parent / (out.name + '_orig') / f'r{r}'; bk.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, bk / path.name)
                meta = dict(meta, damp='0.01+outlier-redamp')
                save_file(tens, str(path) + '.part', metadata=meta); Path(str(path) + '.part').rename(path)
print(json.dumps(dict(event='done', fixed=len(fixed))))
