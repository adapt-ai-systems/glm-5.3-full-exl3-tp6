"""Optional CPU cache preparation, then replace process with unchanged vLLM CLI."""
import fcntl, os, sys, time, traceback
from pathlib import Path
from layout_cache import build_rank, identity

def main():
    started=time.monotonic()
    cache=Path(os.environ.get('TR3_PREPARED_CACHE','/root/.cache/tr3-prepared'))
    rank=int(os.environ['TR3_RANK'])
    if not 0 <= rank < 4: raise ValueError('TP4 rank required')
    try:
        if os.environ.get('TR3_BUILD_PREPARED','1') == '1':
            cache.mkdir(parents=True,exist_ok=True)
            with (cache/f'.build-rank{rank}.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                version=Path('/opt/tr3-load-accel/loader-version.txt').read_text().strip()
                workers=int(os.environ.get('TR3_PREP_WORKERS','2'))
                fast=os.environ.get('TR3_FAST')=='1'
                if fast:  # verification is sha256 of every layer (CPU + disk bound): more threads, same check
                    workers=max(workers,int(os.environ.get('TR3_FAST_HASH_WORKERS','8')))
                import json
                stamp=keep=key0=None
                if fast:  # persistent stat stamp of the last full sha verify (adopted 2026-09-29)
                    try:
                        key0=identity('/model',version)[0];keep=cache/key0/f'rank{rank}'/'.fast-verified.json'
                    except Exception as e:
                        print(f'TR3 fast stat-stamp off rank={rank}: {e!r}',flush=True)
                if keep is not None and os.environ.get('FULLVERIFY','0')!='1' and keep.exists():
                    try:
                        old=json.loads(keep.read_text());key=key0
                        if old['key']==key and old['layers']:
                            for layer,(ino,size,mt,sha) in old['layers'].items():
                                d=cache/key/f'rank{rank}'/f'layer{layer}';st=(d/'tensors.bin').stat()
                                if [st.st_ino,st.st_size,st.st_mtime_ns]!=[ino,size,mt] or json.loads((d/'layout.json').read_text())['sha256']!=sha:
                                    raise ValueError(f'layer {layer} changed')
                            stamp={int(k):v for k,v in old['layers'].items()}
                            print(f'TR3 fast stat-stamp hit rank={rank} layers={len(stamp)} (full sha skipped; FULLVERIFY=1 forces it) seconds={time.monotonic()-started:.3f}',flush=True)
                    except Exception as e:
                        print(f'TR3 fast stat-stamp miss rank={rank}: {e!r} -> full verify',flush=True);stamp=None
                if stamp is None:
                    key=build_rank('/model',cache,rank,version,workers=workers)
                    print(f'TR3 prepared build rank={rank} key={key} workers={workers} seconds={time.monotonic()-started:.3f}',flush=True)
                if fast and stamp is not None:
                    path=f'/tmp/tr3-fast-stamp-rank{rank}-{os.getpid()}.json'
                    with open(path,'w') as f: json.dump(stamp,f)
                    os.environ['TR3_FAST_STAMP']=path
                elif fast:
                    # Still under the build lock: every layer was just sha256-verified (or built) by build_rank.
                    # Record (ino,size,mtime_ns,sha256) so the model-load attach() can trust this boot's check
                    # instead of hashing each layer a second time. pid-unique path -> never reused by a later boot.
                    import json
                    stamp={}
                    for d in sorted((cache/key/f'rank{rank}').glob('layer*')):
                        meta=json.loads((d/'layout.json').read_text());st=(d/'tensors.bin').stat()
                        stamp[int(d.name[5:])]=[st.st_ino,st.st_size,st.st_mtime_ns,meta['sha256']]
                    path=f'/tmp/tr3-fast-stamp-rank{rank}-{os.getpid()}.json'
                    with open(path,'w') as f: json.dump(stamp,f)
                    os.environ['TR3_FAST_STAMP']=path
                    print(f'TR3 fast stamp rank={rank} layers={len(stamp)} {path}',flush=True)
                    if keep is not None and key==key0:  # after a full verify: persist the stamp for the next boot
                      try:
                        tmp=keep.with_suffix('.tmp');tmp.write_text(json.dumps(dict(key=key,layers=stamp)));os.replace(tmp,keep)
                      except OSError as e:
                        print(f'TR3 fast stat-stamp not saved: {e!r}',flush=True)
    except Exception:
        # A partial or incompatible artifact never replaces the original loader.
        traceback.print_exc()
        print('TR3 prepared build failed; falling back to original slow loader',flush=True)
        os.environ['TR3_LOAD_ACCEL']='0'
    os.execvp('vllm',['vllm',*sys.argv[1:]])

if __name__=='__main__': main()
