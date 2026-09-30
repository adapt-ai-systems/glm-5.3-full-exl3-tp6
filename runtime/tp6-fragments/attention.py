"""Storage-only head padding for SM120 sparse MLA's compiled geometries.

Projection/TP heads stay 12 per rank. Only the attention call is padded to
16 (DCP2: 24 ->32); heads are independent, so zero query heads cannot affect
real output heads. Trim output and LSE before any distributed combine.
"""
import torch

def run_sparse_mla_padded(**kwargs):
    from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_with_kv_cache_mla
    query=kwargs['query'];heads=query.shape[-2]
    kernel_heads=next(h for h in (16,32,64,128) if h>=heads)
    if heads==kernel_heads:
        return flashinfer_trtllm_batch_decode_with_kv_cache_mla(**kwargs)
    assert query.ndim==4 and query.shape[1]==1,query.shape
    query_pad=torch.nn.functional.pad(query,(0,0,0,kernel_heads-heads))
    user_out=kwargs['out']
    output=torch.empty((*user_out.shape[:-2],kernel_heads,user_out.shape[-1]),
                       dtype=user_out.dtype,device=user_out.device)
    raw=flashinfer_trtllm_batch_decode_with_kv_cache_mla(**{**kwargs,'query':query_pad,'out':output})
    out,lse=raw if isinstance(raw,tuple) else (raw,None)
    user_out.copy_(out[...,:heads,:])
    if lse is None:return user_out
    # Current FlashInfer runner writes [tokens, heads], checked explicitly.
    if lse.ndim==3 and lse.shape[1]==1:lse=lse[:,0]
    assert lse.shape==(query.shape[0],kernel_heads),lse.shape
    return user_out,lse[:,:heads].contiguous()
