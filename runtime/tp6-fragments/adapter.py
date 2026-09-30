"""GLM6 storage adapters, separate from the original trained dimensions."""
import re
import torch
from fragments import placement

def initialize_layer(layer, moe, num_experts, hidden, intermediate):
    assert moe.moe_parallel_config.tp_size == 6
    assert not moe.moe_parallel_config.use_ep
    assert num_experts == 256 and hidden == 6144 and intermediate == 512
    index = int(re.search(r'layers\.(\d+)\.', str(layer.layer_name))[1])
    pairs = placement(moe.moe_parallel_config.tp_rank, index)
    layer.glm6_pairs = pairs
    layer.local_num_experts = len(pairs)
    return len(pairs)

def extend_route_map(layer, local_map):
    mapping=torch.full((256,),-1,dtype=torch.int32,device=local_map.device)
    ids=torch.tensor([e for e,p in layer.glm6_pairs],dtype=torch.long,device=local_map.device)
    mapping[ids]=local_map
    return mapping

def pad_weight(name, weight):
    """Zero-only padding before standard TP slicing; indexer is untouched."""
    if not name.endswith('.weight') or weight.ndim != 2:return weight
    axis=None;logical=None;physical=None
    if name.endswith('.self_attn.q_b_proj.weight'):
        axis,logical,physical=0,64*256,72*256
    elif name.endswith('.self_attn.kv_b_proj.weight'):
        axis,logical,physical=0,64*(192+256),72*(192+256)
    elif name.endswith('.self_attn.o_proj.weight'):
        axis,logical,physical=1,64*256,72*256
    elif '.mlp.shared_experts.' in name:
        if name.endswith(('.gate_proj.weight','.up_proj.weight')):
            axis,logical,physical=0,2048,2304
        elif name.endswith('.down_proj.weight'):
            axis,logical,physical=1,2048,2304
    if axis is None:return weight
    assert weight.shape[axis] == logical,(name,weight.shape,logical)
    shape=list(weight.shape);shape[axis]=physical
    padded=weight.new_zeros(shape)
    padded.narrow(axis,0,logical).copy_(weight)
    return padded
