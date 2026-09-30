"""Anchor-checked build patch, staged separately from measured fastload-v1."""
from pathlib import Path
import sys


def patch_source(source):
    old = '''from vllm.model_executor.model_loader.ep_weight_filter import (
    should_skip_weight,
)'''
    new = '''from vllm.model_executor.model_loader.ep_weight_filter import (
    should_skip_weight as _glm6_original_skip_weight,
)
from fastload_filter import wrap_skip_weight as _glm6_wrap_skip_weight
should_skip_weight = _glm6_wrap_skip_weight(_glm6_original_skip_weight)'''
    if source.count(old) != 1:
        raise ValueError('Expected exactly one pinned safetensors EP-filter import')
    # These are the three checked iterator branches: eager, torchao and lazy.
    if source.count('should_skip_weight(name, local_expert_ids)') != 3:
        raise ValueError('Pinned safetensors iterator changed; inspect before patching')
    result = source.replace(old, new)
    compile(result, '<glm6-weight-utils>', 'exec')
    return result


if __name__ == '__main__':
    root = Path(sys.argv[1] if len(sys.argv) > 1 else
                '/usr/local/lib/python3.12/dist-packages/vllm')
    path = root / 'model_executor/model_loader/weight_utils.py'
    path.write_text(patch_source(path.read_text()))
    print('GLM6 redundant raw expert iteration filtered; direct-tier loader unchanged')
