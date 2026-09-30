"""Opt-in loader hooks. Imported only by the generated EXL3 variant."""
import json, logging, os, re, time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from layout_cache import identity, validate

log=logging.getLogger('vllm.tr3_load_accel')

# ---- TP4 fastload (TR3_FAST=1). Off -> every function below behaves exactly like the original.
FAST=os.environ.get('TR3_FAST')=='1'
CONSTRUCTED=set()   # layer indices whose EXL3 MoE was constructed in this process (attach seen)
HIT=set()           # layer indices restored from the prepared cache (attach HIT)
RESTORED=set()      # layer indices restore() completed


@lru_cache(maxsize=1)
def _stamp():
    """{layer: (ino,size,mtime_ns,sha256)} written by the fast entrypoint after it sha256-verified every layer
    of this rank in THIS boot (path passed by env, pid-unique). Missing/unreadable -> {} (full hash as before)."""
    p=os.environ.get('TR3_FAST_STAMP')
    if not (FAST and p): return {}
    try:
        return {int(k):tuple(v) for k,v in json.loads(Path(p).read_text()).items()}
    except (OSError,ValueError,TypeError):
        log.exception('TR3 fast: stamp unreadable; full validation')
        return {}

@lru_cache(maxsize=1)
def settings():
    if os.environ.get('TR3_LOAD_ACCEL') != '1': return None
    model=Path(os.environ.get('TR3_CACHE_MODEL','/model'))
    version=Path(os.environ.get('TR3_LOADER_VERSION_FILE','/opt/tr3-load-accel/loader-version.txt')).read_text().strip()
    key,_=identity(model,version)
    return Path(os.environ.get('TR3_PREPARED_CACHE','/root/.cache/tr3-prepared'))/key,key


def attach(layer):
    if not (getattr(layer,'exl3_rank_sliced',False) and getattr(layer,'exl3_mixed_bitrate',False)):
        return
    try:
        s=settings()
        if s is None: return
        root,key=s
        m=re.search(r'layers\.(\d+)\.',str(layer.layer_name))
        if m is None: return
        index=int(m[1]);rank=int(layer.exl3_tp_rank)
        CONSTRUCTED.add(index)
        directory=root/f'rank{rank}'/f'layer{index}'
        trusted=False
        if FAST and index in _stamp():
            st=(directory/'tensors.bin').stat()
            trusted=_stamp()[index][:3]==(st.st_ino,st.st_size,st.st_mtime_ns)
        meta=validate(directory,key=key,rank=rank,layer=index,bits=layer.exl3_layer_bitrates,
                      hidden=layer.exl3_hidden_size,intermediate=layer.exl3_intermediate_size_per_partition,
                      checksum=not trusted)
        if meta is not None and trusted and meta.get('sha256')!=_stamp()[index][3]:
            meta=validate(directory,key=key,rank=rank,layer=index,bits=layer.exl3_layer_bitrates,
                          hidden=layer.exl3_hidden_size,intermediate=layer.exl3_intermediate_size_per_partition)
        if meta is None:
            log.info('TR3 prepared cache MISS rank=%s layer=%s; original loader fallback',rank,index)
            return
        layer._tr3_cache_entry=(directory,meta,(directory/'tensors.bin').stat())
        if FAST:
            HIT.add(index);_prefetch_first(directory,meta)
        for prefix in ('w13','w2'):
            for suffix in ('suh','svh','trellis','mcg','mul1'):
                getattr(layer,f'{prefix}_{suffix}')._tr3_cache_skip=True
        log.info('TR3 prepared cache HIT rank=%s layer=%s%s',rank,index,' (stamp-verified this boot)' if trusted else '')
    except (OSError,ValueError,KeyError,TypeError):
        log.exception('TR3 cache identity/validation failed; original loader fallback')


def restore(method,layer,mixed_api):
    entry=getattr(layer,'_tr3_cache_entry',None)
    if entry is None: return False
    import numpy as np
    import torch
    started=time.monotonic()
    directory,meta,oldstat=entry
    stat=(directory/'tensors.bin').stat()
    if (stat.st_ino,stat.st_size,stat.st_mtime_ns)!=(oldstat.st_ino,oldstat.st_size,oldstat.st_mtime_ns):
        raise RuntimeError('TR3 prepared file changed after validation; refusing partial cached load')
    device=layer.w13_trellis.device
    arrays={}
    if FAST:
        arrays=_fast_arrays(directory,meta,device,torch,oldstat)
    for name,e in ({} if FAST else meta['tensors']).items():
        # File-backed CPU mapping; explicitly copy into owning CUDA allocation.
        # copy-on-write mapping is writable to numpy/torch but never dirtied here.
        host=np.memmap(directory/'tensors.bin',mode='c',dtype=e['dtype'],
                       offset=e['offset'],shape=tuple(e['shape']))
        arrays[name]=torch.from_numpy(host).to(device=device,copy=True)
        del host
    hidden=int(layer.exl3_hidden_size);intermediate=int(layer.exl3_intermediate_size_per_partition)
    configs=(method._mixed_trellis_tile_config(hidden,intermediate),
             method._mixed_trellis_prefill_tile_config(hidden,intermediate))
    bits=list(layer.exl3_layer_bitrates)
    tiers=[(k,tuple(e for e,b in enumerate(bits) if b==k)) for k in sorted(set(bits))]
    prepared=[[],[]];offset=0
    for k,ids in tiers:
        sl=slice(offset,offset+len(ids));w13=arrays[f'w13_k{k}'];w2=arrays[f'w2_k{k}']
        for objects,config in zip(prepared,configs):
            objects.append(mixed_api.prepare_weights(w13=w13,w2=w2,hidden_size=hidden,
                intermediate_size=intermediate,num_experts=len(ids),activation=layer.activation.value,
                fc1_tile_n=config[1],fc2_tile_n=config[3],params_dtype=torch.float16,
                w13_layout='trellis_t256_proj',trellis_bits=k,codebook='mcg',
                gate_suh=arrays['gate_suh'][sl],up_suh=arrays['up_suh'][sl],
                intermediate_rotations=arrays['intermediate'][sl],down_svh=arrays['down_svh'][sl],
                tile_config=config,workspace=w13.view(torch.int32).reshape(-1)[:1]))
        offset+=len(ids)
    counts=tuple(len(ids) for _,ids in tiers)
    arrays['descriptor_map']._mt_projection_counts=(counts,counts)
    layer.exl3_mixed_trellis=dict(tiers=tuple(prepared[0]),prefill_tiers=tuple(prepared[1]),
        tier_ids=tuple(ids for _,ids in tiers),tier_bits=tuple(k for k,_ in tiers),
        trellis_codebook='mcg',global_to_combined=arrays['global_to_combined'],
        descriptor_map=arrays['descriptor_map'],rotations=SimpleNamespace(
            intermediate=arrays['intermediate'],gate_suh=arrays['gate_suh'],
            up_suh=arrays['up_suh'],down_svh=arrays['down_svh']),
        broadcast_suh=False,broadcast_svh=False,tile_config=configs[0],prefill_tile_config=configs[1])
    layer.exl3_trellis_tile_config=configs[0]
    if FAST:
        import sys
        index=int(re.search(r'layers\.(\d+)\.',str(layer.layer_name))[1])
        f=sys.modules.get('vllm.model_executor.model_loader.ep_weight_filter')
        counts=getattr(f,'_tr3_fast_counts',None)
        if counts is not None and (not RESTORED or index==78):  # first restore = target pass done; 78 = drafter pass done
            log.info('TR3 fast filter skipped so far: %s',counts)
        RESTORED.add(index)
    log.info('TR3 prepared restore %s %.3fs (%s -> device, no retiering)',layer.layer_name,time.monotonic()-started,
             'O_DIRECT pool' if FAST else 'mmap')
    return True


# ---- fast reader: parallel O_DIRECT reads of tensors.bin into a page-aligned host buffer, the next HIT layer
# prefetched in the background while this one is copied/prepared. Bytes are identical to the mmap path (same file,
# same offsets); the file's stat is re-checked against the attach-time stat by restore() before and here after reading.
import threading
from concurrent.futures import ThreadPoolExecutor
_CHUNK=16<<20
_pool=None
_sub=None
_lock=threading.Lock()
_pending={}   # directory -> future of its read buffer
_order=[]     # HIT directories in attach order (restore follows construction order)
_metas={}


def _workers():
    return max(1,min(16,int(os.environ.get('TR3_FAST_READERS','8'))))


def _get_pool():
    """_pool runs whole-file reads (at most 2 in flight: current + next layer); _sub runs their chunks."""
    global _pool,_sub  # called with _lock held (from _submit)
    if _pool is None:
        _pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='tr3fast')
        _sub=ThreadPoolExecutor(max_workers=_workers(),thread_name_prefix='tr3fast-io')
    return _pool


def _read_file(path,nbytes):
    """Whole file -> anonymous page-aligned buffer (mmap), 16 MiB chunks in parallel. O_DIRECT if the fs allows it
    (no page-cache churn under unified-memory pressure), plain pread otherwise. Returns the mmap buffer (caller closes)."""
    import mmap
    size=(nbytes+4095)//4096*4096
    buf=mmap.mmap(-1,max(size,4096))
    try:
        fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECT',0))
        direct=True
    except OSError:
        fd=os.open(path,os.O_RDONLY);direct=False
    try:
        view=memoryview(buf)
        def chunk(off):
            n=min(_CHUNK,size-off);got=0
            while got<n:
                try:
                    r=os.preadv(fd,[view[off+got:off+n]],off+got)
                except OSError:
                    if not direct: raise
                    fd2=os.open(path,os.O_RDONLY)
                    try: r=os.preadv(fd2,[view[off+got:off+n]],off+got)
                    finally: os.close(fd2)
                if r==0: break
                got+=r
            return off+got if got<n else None
        ends=[f.result() for f in [_sub.submit(chunk,o) for o in range(0,size,_CHUNK)]]
        ends=[e for e in ends if e is not None]
        short=[e for e in ends if e<nbytes]
        if short: raise OSError(f'TR3 fast: short read of {path} ({min(short)} < {nbytes})')
        return buf
    finally:
        os.close(fd)



def _submit(directory,meta):
    with _lock:
        if directory in _pending: return
        _pending[directory]=_get_pool().submit(_read_file,directory/'tensors.bin',int(meta['nbytes']))


def _prefetch_first(directory,meta):
    with _lock:
        _order.append(directory);_metas[directory]=meta
    if len(_order)==1: _submit(directory,meta)


def _next_after(directory):
    with _lock:
        if directory in _order:
            i=_order.index(directory)
            if i+1<len(_order): return _order[i+1]
    return None


def _fast_arrays(directory,meta,device,torch,oldstat):
    _submit(directory,meta)                 # no-op if prefetched
    with _lock: fut=_pending.pop(directory)
    buf=fut.result()
    st=(directory/'tensors.bin').stat()
    if (st.st_ino,st.st_size,st.st_mtime_ns)!=(oldstat.st_ino,oldstat.st_size,oldstat.st_mtime_ns):
        buf.close()
        raise RuntimeError('TR3 prepared file changed during fast read; refusing partial cached load')
    nxt=_next_after(directory)
    if nxt is not None: _submit(nxt,_metas[nxt])  # overlap the next layer's disk read with this layer's copies
    try:
        arrays={}
        import numpy as np
        for name,e in meta['tensors'].items():
            host=np.frombuffer(buf,dtype=e['dtype'],count=int(np.prod(e['shape'])),offset=e['offset']).reshape(tuple(e['shape']))
            arrays[name]=torch.from_numpy(host).to(device=device,copy=True)
            del host
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        return arrays
    finally:
        try: buf.close()
        except BufferError: pass
