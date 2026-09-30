"""Opt-in TP4 apply-only hook; native loader and prepared-cache bytes stay intact."""
from functools import wraps
import hashlib
import importlib.abc
import json
import logging
import os
from pathlib import Path
import sys

TARGET = 'vllm.model_executor.layers.quantization.exl3'
LOG = logging.getLogger('vllm.tp4_e3')
PROVENANCE = json.loads(Path(__file__).with_name('provenance.json').read_text())
D2_OVERLAY_SHA256 = 'a098b610825b142d9caf1969719396d17d3bb96f15b9307a002afcb55939bfa7'  # overlay/exl3.py


def validate_layer(layer):
    """Metadata checks only: no device-to-host copies in apply/capture."""
    mixed = layer.exl3_mixed_trellis
    signature = tuple((int(bits), len(ids)) for bits, ids in
                      zip(mixed['tier_bits'], mixed['tier_ids'], strict=True))
    if (int(layer.exl3_hidden_size) != 6144
            or int(layer.exl3_intermediate_size_per_partition) != 512
            or int(layer.local_num_experts) != 256
            or signature != ((3, 192), (4, 64))
            or mixed['trellis_codebook'] != 'mcg'
            or tuple(mixed['global_to_combined'].shape) != (256,)
            or int(layer.exl3_max_num_batched_tokens) not in (2048, 4096)):
        raise ValueError('TP4 E3 requires h6144/i512, K3:192/K4:64, capacity2048 or4096')
    # The native loader constructs global_to_combined in this exact tier order.
    # Check the CPU ids rather than synchronizing a CUDA mapping tensor.
    ids = [int(e) for tier in mixed['tier_ids'] for e in tier]
    if sorted(ids) != list(range(256)):
        raise ValueError('TP4 E3 requires every logical expert exactly once')


def install(module):
    """Wrap only execution, after normal module import and source verification."""
    source = Path(module.__file__)
    native = PROVENANCE['native_loader_sha256']['load_accel/exl3_cached.py']
    # tp4-speed-20260927: the D2 overlay exl3.py is exl3_cached.py + a 4-line dense get_quant_method hook;
    # the MoE apply path this module wraps is byte-identical, so the D2 overlay sha is also accepted.
    allowed = {native, D2_OVERLAY_SHA256}
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected not in allowed:
        raise RuntimeError('TP4 E3 native EXL3 source mismatch; restage against actual loader')
    cls = module.Exl3MoEMethod
    original = cls._apply_mixed_rank_sliced
    if getattr(original, '_tp4_e3', False):
        return
    from . import serving

    @wraps(original)
    def apply(self, layer, x, topk_weights, topk_ids):
        if os.environ.get('TP4_E3_PREFILL', '0') != '1' or int(x.shape[0]) <= 32:
            return original(self, layer, x, topk_weights, topk_ids)
        if not serving.can_apply(layer, x, topk_ids):
            return original(self, layer, x, topk_weights, topk_ids)
        if not hasattr(layer, 'glm6_e3_binding'):
            validate_layer(layer)
        # Mandatory route-pack warmup needs native B12X planning, even on E3.
        self._mixed_rank_sliced_runtime(layer, x, topk_ids)
        return serving.apply(layer, x, topk_weights, topk_ids)

    apply._tp4_e3 = True
    cls._apply_mixed_rank_sliced = apply
    LOG.info('TP4_E3_INSTALLED native_sha256=%s allowed_capacities=2048,4096 stream_scratch=True', expected)


class _Loader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        install(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        # Delegate to the existing finder order, including any custom importer.
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            raise ImportError('TP4 E3 could not locate native EXL3 module')
        spec.loader = _Loader(spec.loader)
        return spec


def register():
    if os.environ.get('TP4_E3_PREFILL', '0') != '1':
        return False
    # No CUDA, torch, vLLM or checkpoint access during Python startup.
    for name, expected in PROVENANCE['copied_sha256'].items():
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'TP4 E3 artifact mismatch: {name}')
    if TARGET in sys.modules:
        install(sys.modules[TARGET])
    elif not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    return True
