"""model-layer-LLL.safetensors -> dense/layer-LLL.safetensors with only the D8 GPTQ targets' BF16 sources
(q_a_proj, o_proj, [shared_experts.]gate/up/down). Usage: extract_dense.py <in> <out>"""
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

KEEP = r'model\.layers\.\d+\.(self_attn\.(q_a_proj|o_proj)|mlp\.(shared_experts\.)?(gate_proj|up_proj|down_proj))\.weight'
with safe_open(sys.argv[1], 'pt') as f:
    t = {k: f.get_tensor(k) for k in f.keys() if re.fullmatch(KEEP, k)}
assert len(t) in (5,), (sys.argv[1], sorted(t))
save_file(t, sys.argv[2])
print(sys.argv[2], len(t), sum(v.numel() for v in t.values()) * 2 / 1e9, 'GB')
