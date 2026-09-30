"""Thin-image patch; only changes startup loading, never forward kernels."""
from pathlib import Path
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py')
s=p.read_text()
old='        # Do not attach the incompatible old TP4 prepared cache.'
assert s.count(old)==1
s=s.replace(old,'        from fastload import attach as glm6_attach\n        glm6_attach(layer)')
old='    def process_weights_after_loading(self, layer: RoutedExperts) -> None:\n'
assert s.count(old)==1
s=s.replace(old,old+'        from fastload import restore as glm6_restore\n        if glm6_restore(self, layer, _load_b12x_mixed_trellis()):\n            return\n')
compile(s,str(p),'exec');p.write_text(s)
print('GLM6 direct-tier loader enabled; original path retained when GLM6_FAST_LOAD=0')
