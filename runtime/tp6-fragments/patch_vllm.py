"""Build-time, anchor-checked patch of the pinned Mia + TR3 runtime."""
from pathlib import Path
import sys,re
root=Path(sys.argv[1] if len(sys.argv)>1 else '/usr/local/lib/python3.12/dist-packages/vllm')
def replace(s,old,new,count=1):
    assert s.count(old)==count,(old[:130],s.count(old),count)
    return s.replace(old,new)

for module in ('deepseek_v2','deepseek_mtp'):
    p=root/'model_executor/models'/f'{module}.py';s=p.read_text()
    s=replace(s,'        for name, loaded_weight in weights:\n',
        '        from adapter import pad_weight\n        for name, loaded_weight in weights:\n            loaded_weight = pad_weight(name, loaded_weight)\n')
    # Original logical vocabulary: logits processing strips storage padding
    # before probabilities, sampling and MTP checks.
    for constructor in ('VocabParallelEmbedding','ParallelLMHead'):
        pattern=rf'(= {constructor}\(\n)\s+(config.vocab_size,\n)( +)((?:config.hidden_size|self.hidden_size),\n)'
        def insert(m):
            return m[1]+m[3]+m[2]+m[3]+m[4]+m[3]+'padding_size=384,\n'
        s,n=re.subn(pattern,insert,s);assert n==1,(module,constructor,n)
    if module=='deepseek_v2':
        s=replace(s,'intermediate_size = config.moe_intermediate_size * config.n_shared_experts',
            'intermediate_size = 2304 * config.n_shared_experts  # 2048 real + zero channels')
        s=replace(s,'            intermediate_size=config.moe_intermediate_size,',
            '            intermediate_size=3072,  # six local 512 slots; four fragments/expert')
    compile(s,str(p),'exec');p.write_text(s)

p=root/'model_executor/layers/quantization/exl3.py';s=p.read_text()
s=replace(s,'        """Drop non-local TP payloads and remove the serialized rank segment."""',
    '        """Map original expert fragments into the six-rank local namespace."""\n        from fragments import normalize\n        return normalize(name, get_tensor_model_parallel_rank())\n\n        # Original TP4 normalization below is unreachable in this image.')
anchor='''        del extra_weight_attrs
        if params_dtype not in (torch.bfloat16, torch.float16):'''
s=replace(s,anchor,'''        del extra_weight_attrs
        from adapter import initialize_layer
        num_experts = initialize_layer(layer, self.moe, num_experts, hidden_size,
                                       intermediate_size_per_partition)
        if params_dtype not in (torch.bfloat16, torch.float16):''')
s=replace(s,'            if checkpoint_tp != layer.exl3_tp_size:',
    '            if checkpoint_tp != 4 or layer.exl3_tp_size != 6:')
s=replace(s,'            if expected_experts != num_experts:',
    '            if expected_experts != 256:')
anchor='''            layer.exl3_layer_bitrates = self.quant_config.rank_sliced_layer_bitrates(
                str(layer.layer_name)
            )'''
s=replace(s,anchor,anchor+'''
            layer.exl3_layer_bitrates = tuple(
                layer.exl3_layer_bitrates[e] for e, _ in layer.glm6_pairs)''')
s=replace(s,'        from tr3_runtime import attach as tr3_attach\n        tr3_attach(layer)',
    '        # Do not attach the incompatible old TP4 prepared cache.')
anchor='''        global_to_combined, descriptor_map = mixed_api.build_tiered_maps(
            tier_ids[0], tier_ids[1], device=device
        )'''
s=replace(s,anchor,anchor+'''
        from adapter import extend_route_map
        global_to_combined = extend_route_map(layer, global_to_combined)''')
s=replace(s,'        route_num_experts = int(layer.local_num_experts)',
    '        route_num_experts = int(mixed["global_to_combined"].numel())')
compile(s,str(p),'exec');p.write_text(s)
print('GLM6 runtime patch applied')

# Use the already-published DCP-correct backend, with call-local head padding.
p=root/'v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py'
s=p.read_text()
s=replace(s,"        from vllm.utils.flashinfer import (\n            flashinfer_trtllm_batch_decode_with_kv_cache_mla,\n        )",
    "        from attention import run_sparse_mla_padded as flashinfer_trtllm_batch_decode_with_kv_cache_mla")
compile(s,str(p),'exec');p.write_text(s)
