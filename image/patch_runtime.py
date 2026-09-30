"""Minimal bridge from the Mia ARM runtime to the upstream r17 TR3 loader."""
from pathlib import Path
import sys
root = Path(sys.argv[1] if len(sys.argv) > 1 else '/usr/local/lib/python3.12/dist-packages/vllm')
for name in ('deepseek_v2', 'deepseek_mtp'):
    path = root / 'model_executor/models' / (name + '.py')
    source = path.read_text()
    anchor = '        for name, loaded_weight in weights:\n'
    assert source.count(anchor) == 1, (name, source.count(anchor))
    # The target inner model does not retain quant_config; the MTP model does.
    if name == 'deepseek_v2':
        old = '        self.config = config\n        self.device = current_platform.device_type'
        assert source.count(old) == 1
        source = source.replace(old, '        self.quant_config = quant_config\n' + old)
    source = source.replace(anchor, anchor + '''            # TR3 payloads are already TP4-sharded: discard foreign ranks,
            # remove only the .rankN component, and never slice them again.
            normalize = getattr(self.quant_config, "normalize_rank_sliced_weight_name", None)
            if normalize is not None:
                name = normalize(name)
                if name is None:
                    continue
''')
    path.write_text(source)
# This exact full-model prefill fix is published by drowzeys/keys-GLM-5.3-EXL3.
# Persistent topk's 2048-token configuration oversubscribes the 48-SM GB10.
path = root / 'model_executor/layers/sparse_attn_indexer.py'
s = path.read_text()
old = 'use_persistent_topk = current_platform.is_cuda() and topk_tokens in ('
assert s.count(old) == 1, s.count(old)
path.write_text(s.replace(old, 'use_persistent_topk = False and current_platform.is_cuda() and topk_tokens in ('))
# Pure path predicate from the pinned upstream r17 base (absent in Mia's tree).
path = root / 'model_executor/layers/quantization/online/mxfp8.py'
s = path.read_text()
assert 'def is_shared_expert_projection(' not in s
s += '''\n\n# r17 compatibility: projection ownership, no numerical changes.
def is_shared_expert_projection(prefix: str) -> bool:
    parts = prefix.split(".")
    return (len(parts) >= 2
            and parts[-1] in {"gate_proj", "up_proj", "gate_up_proj", "down_proj"}
            and parts[-2] in {"shared_expert", "shared_experts"})
'''
path.write_text(s)
