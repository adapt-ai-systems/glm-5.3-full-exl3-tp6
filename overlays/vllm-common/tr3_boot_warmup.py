"""Compile/materialize TR3 MTP and prefix/chunk metadata specializations at boot.
No model requests, no kernel launches, and no mutation of serving tensors.
"""
from __future__ import annotations
import itertools, time
from collections import Counter
import torch
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor

logger=init_logger(__name__)


def chunk_pairs(max_tokens):
    # Triton non-constexpr integers specialize equal-to-one and alignment.
    # Include nonzero nonaligned chunk starts, absent from stock start={0,1}.
    starts=(0,1,2,15,16,17)
    stops=sorted({1,2,15,16,17,32,max(1,max_tokens-1),max_tokens})
    return tuple((a,b) for a in starts for b in stops if b>a)


def pointer(dtype,aligned=True):
    return TritonWarmupTensor(dtype,aligned=aligned)


def compile_plan(worker):
    """Yield exact current-runtime argument signatures; CPU-testable with stubs."""
    from vllm.v1.attention.backends.mla.indexer import (
        _BUILD_PREFILL_CHUNK_METADATA_KERNEL, _prepare_uniform_decode_kernel)
    from vllm.v1.spec_decode.utils import (
        eagle_prepare_inputs_padded_kernel, eagle_prepare_next_token_padded_kernel,
        eagle_step_slot_mapping_metadata_kernel)
    from vllm.distributed import get_dcp_group
    cfg=worker.vllm_config;runner=worker.model_runner
    p=pointer(torch.int32)
    parallel=cfg.parallel_config;world=parallel.decode_context_parallel_size
    rank=get_dcp_group().rank_in_group if world>1 else 0
    ratios=set(getattr(cfg.model_config.hf_config,'compress_ratios',None) or [1])
    ratios.add(max(1,int(getattr(cfg.model_config.hf_config,'index_kpool',1) or 1)))
    for (start,stop),aligned,ratio in itertools.product(
            chunk_pairs(cfg.scheduler_config.max_num_batched_tokens),(True,False),sorted(ratios)):
        args=(p,pointer(torch.int32,aligned),p,p,p,p,p,start,stop,rank,world,
              parallel.cp_kv_cache_interleave_size)
        yield _BUILD_PREFILL_CHUNK_METADATA_KERNEL.kernel,args,dict(
            BLOCK_SIZE=_BUILD_PREFILL_CHUNK_METADATA_KERNEL.BLOCK_SIZE,
            COMPRESS_RATIO=int(ratio),grid=(1,))
    spec=cfg.speculative_config
    if spec is None:return
    n_spec=int(spec.num_speculative_tokens)
    reqs=range(1,cfg.scheduler_config.max_num_seqs+1)
    # Prefill / zero-draft row and padded verification width; intermediate
    # accepted counts are data, but include all legal row widths up to n+1.
    for n,width,dtype in itertools.product(reqs,range(1,n_spec+2),(torch.int32,torch.int64)):
        for aligned in (True,False):
            yield eagle_prepare_next_token_padded_kernel,(
                pointer(dtype,aligned),pointer(torch.bool),p,p,p,
                runner.input_batch.vocab_size,width,n,width),dict(
                BLOCK_SIZE_TOKENS=1<<(width-1).bit_length(),grid=(n,))
    for n,aligned in itertools.product(reqs,(True,False)):
        yield eagle_prepare_inputs_padded_kernel,(pointer(torch.int32,aligned),p,p,p,p,n),dict(grid=(n,))
    tables=runner.input_batch.block_table.block_tables
    drafter=runner.drafter
    position_dtype=drafter.positions.dtype
    slot_dtype=drafter._slot_mapping_buffer.dtype
    for table in tables:
        tensor=table.get_device_tensor(1)
        stride=tensor.stride(0);columns=tensor.shape[1]
        for n,aligned in itertools.product(reqs,(True,False)):
            yield eagle_step_slot_mapping_metadata_kernel,(
                pointer(position_dtype,aligned),p,stride,p,pointer(position_dtype),
                pointer(slot_dtype),),dict(block_size=table.block_size,
                max_model_len=drafter.max_model_len,n_blocks_per_req=columns,
                PAD_ID=-1,batch_size=n,grid=(n,))
    # Target and drafter indexer builders can use different expanded table widths.
    groups=[g for row in runner.attn_groups for g in row]
    groups+=list(getattr(drafter,'draft_attn_groups',[]))
    widths=set()
    for group in groups:
        builder=group.get_metadata_builder()
        expanded=getattr(builder,'expanded_block_table_buffer',None)
        if expanded is not None:widths.add(expanded.stride(0))
    if not widths:
        raise RuntimeError('TR3 warmup found no initialized indexer decode buffers')
    for table,width,max_decode,aligned in itertools.product(tables,sorted(widths),range(1,n_spec+2),(True,False)):
        stride=table.get_device_tensor(1).stride(0)
        yield _prepare_uniform_decode_kernel,(
            pointer(torch.int32,aligned),p,p,stride,p,width,p,max_decode),dict(BLOCK_SIZE=1024,grid=(max_decode,))


def warmup_tr3_metadata(worker):
    """Two passes: disk/compiler diagnostics, then zero-compile in-process replay.

    .warmup() alone leaves CompiledKernel CUDA handles lazy. Materialize those
    handles too, without executing a metadata kernel against live buffers.
    """
    from triton import knobs
    started=time.monotonic();plan=list(compile_plan(worker));counts=Counter()
    previous=knobs.compilation.listener
    def listener(**kwargs):
        counts['disk_hit' if kwargs.get('cache_hit') else 'compiled']+=1
        if previous is not None:previous(**kwargs)
    knobs.compilation.listener=listener
    try:
        for phase in ('first','replay'):
            counts.clear();t=time.monotonic()
            for kernel,args,kwargs in plan:
                artifact=kernel.warmup(*args,**kwargs)
                if artifact is None:
                    raise RuntimeError('TR3 metadata warmup requires synchronous Triton compilation')
                artifact._init_handles()
            logger.info('TR3 metadata warmup phase=%s signatures=%d compiled=%d disk_hit=%d seconds=%.3f cache=%s',
                phase,len(plan),counts['compiled'],counts['disk_hit'],time.monotonic()-t,knobs.cache.dir)
            if phase=='replay' and counts['compiled']:
                raise RuntimeError('TR3 metadata warmup replay unexpectedly recompiled kernels')
    finally:
        knobs.compilation.listener=previous
    logger.info('TR3 metadata warmup complete in %.3fs (MTP/prefix/chunk signatures, CUDA handles loaded)',time.monotonic()-started)
