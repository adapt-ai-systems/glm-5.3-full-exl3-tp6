"""E3-v2 control-plane adapter: tested stream arenas, optional one-shot audit.

No new kernel/grid/arithmetic. Audit is explicitly armed AFTER ready by writing
/tmp/e3-audit/ARM. Disarm after the diagnostic request: the next layer3
call permanently disables audit for clean timing on this boot. No tensor dump
or completion-event creation occurs outside the armed window. Event completion
is OBSERVED later; log UTC is not asserted to be the device finish timestamp.
"""
import json,logging,os,time
from pathlib import Path
from . import runtime

LOG=logging.getLogger('vllm.glm6_e3_serving')
_DIR=Path('/tmp/e3-audit')
_ACTIVE=None
_DISABLED=os.environ.get('GLM6_E3_AUDIT','0')!='1'
_EVER_ARMED=False
_PENDING=[]
_DUMPED=set()
_SEEN=set()
_SEQ=0


def key_for(layer,x,ids,stream):
    return (x.device.index,int(x.shape[1]),int(layer.exl3_intermediate_size_per_partition),
            int(layer.exl3_max_num_batched_tokens),int(ids.shape[1]),stream)


def can_apply(layer,x,ids):
    """On a cold graph stream select B12X before ANY E3 launch/allocation."""
    import torch
    if not torch.cuda.is_current_stream_capturing():return True
    stream=torch.cuda.current_stream(x.device).cuda_stream
    ready=(hasattr(layer,'glm6_e3_binding') and x.device.index in runtime._MODULES
           and key_for(layer,x,ids,stream) in runtime._SCRATCH)
    if not ready:LOG.info('GLM6_E3_CAPTURE_FALLBACK layer=%s m=%d stream=%d',layer.layer_name,x.shape[0],stream)
    return ready


def emit(kind,**data):
    _DIR.mkdir(exist_ok=True)
    with (_DIR/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind,observed_epoch=time.time(),pid=os.getpid(),**data))+'\n')


def flush_completed():
    # Query is nonblocking. Keep a record of unresolved GPU completion rather
    # than silently equating HOST_RETURN with execution completion.
    pending=[]
    for event,meta in _PENDING:
        if event.query():emit('DEVICE_COMPLETED_OBSERVED',**meta)
        else:pending.append((event,meta))
    _PENDING[:]=pending


def apply(layer,x,weights,ids):
    global _ACTIVE,_DISABLED,_EVER_ARMED,_SEQ
    import torch
    stream=torch.cuda.current_stream(x.device).cuda_stream
    capture=torch.cuda.is_current_stream_capturing()
    if _PENDING and not capture:flush_completed()
    # Only one selected layer polls the arm marker; after disarm no filesystem
    # polling or events in the timed path. Never dump inside CUDA capture.
    if not _DISABLED and layer.layer_name=='model.layers.3.mlp.experts' and not capture:
        try:tag=(_DIR/'ARM').read_text().strip()
        except FileNotFoundError:tag=None
        if tag:_ACTIVE=tag;_EVER_ARMED=True
        elif _EVER_ARMED:
            _ACTIVE=None;_DISABLED=True;emit('AUDIT_DISABLED_FOR_TIMING',pending_completions=len(_PENDING))
    k=key_for(layer,x,ids,stream)
    first=k not in _SEEN
    if first:
        LOG.info('GLM6_E3_STREAM_FIRST layer=%s m=%d stream=%d capture=%s arenas_before=%d',
                 layer.layer_name,x.shape[0],stream,capture,len(runtime._SCRATCH))
    meta=None
    if _ACTIVE and not capture:
        _SEQ+=1
        meta=dict(tag=_ACTIVE,seq=_SEQ,layer=layer.layer_name,rows=int(x.shape[0]),stream=stream,capture=False,
                  rank=int(layer.exl3_tp_rank),arenas_before=len(runtime._SCRATCH))
        emit('HOST_ENTER',**meta)
        if layer.layer_name=='model.layers.3.mlp.experts' and x.shape[0]==4096 and _ACTIVE not in _DUMPED:
            # This CPU transfer synchronizes an explicit diagnostic request.
            # It is excluded from all clean timing claims.
            torch.save(dict(x=x.detach().cpu(),weights=weights.detach().cpu(),ids=ids.detach().cpu(),
                meta=meta,scope='armed real request; synchronous fixture copy, not timed'),
                _DIR/('route-fixture-r'+str(layer.exl3_tp_rank)+'.pt'))
            routes=runtime.route_tables(ids,weights,layer.exl3_mixed_trellis['global_to_combined'],sum(len(t) for t in layer.exl3_mixed_trellis['tier_ids']))
            emit('REAL_ROUTES',**meta,actual_rows=int(routes['num_rows'].item()),actual_segments=int(routes['num_segs'].item()),
                 padded_segments=routes['seg_expert'].numel(),ids_min=int(ids.min().item()),ids_max=int(ids.max().item()))
            del routes;_DUMPED.add(_ACTIVE)
    out=runtime.apply(layer,x,weights,ids,stream_scratch=True)
    if first:
        _SEEN.add(k)
        allocated=sum(t.numel()*t.element_size() for arena in runtime._SCRATCH.values() for t in arena.values())
        LOG.info('GLM6_E3_ARENAS layer=%s m=%d stream=%d capture=%s count=%d bytes=%d torch_allocated=%d',
                 layer.layer_name,x.shape[0],stream,capture,len(runtime._SCRATCH),allocated,torch.cuda.memory_allocated())
    if meta:
        event=torch.cuda.Event();event.record();_PENDING.append((event,meta))
        emit('HOST_RETURN_GPU_PENDING',**meta,arenas_after=len(runtime._SCRATCH))
    return out
