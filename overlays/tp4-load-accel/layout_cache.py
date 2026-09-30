"""CPU-only, mmap-backed deterministic TR3 tier-layout cache. No torch import."""
from __future__ import annotations
import concurrent.futures, hashlib, json, mmap, os, re, shutil, struct, tempfile
from pathlib import Path
import numpy as np

SCHEMA = 'tr3-native-tiers-v1'
PATTERN = re.compile(r'model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.rank(\d+)\.(trellis|suh|svh|mcg)$')
DTYPES = {'I16': '<i2', 'F16': '<f2', 'I32': '<i4', 'U32': '<u4'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def identity(model, loader_version):
    model = Path(model)
    obj = dict(schema=SCHEMA, manifest=digest(model/'MANIFEST.sha256'),
               config=digest(model/'config.json'), tiers=digest(model/'tier_bitmap.json'),
               loader_version=loader_version)
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest(), obj


def index_rank(model, rank):
    result = {}
    for path in sorted(Path(model).glob('*.safetensors')):
        with path.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            if n > 256 << 20: raise ValueError('oversized safetensors header')
            header = json.loads(f.read(n))
        for name, meta in header.items():
            m = PATTERN.fullmatch(name)
            if m is None or int(m[4]) != rank: continue
            layer, expert = int(m[1]), int(m[2])
            key = (expert, m[3], m[5])
            layer_index = result.setdefault(layer, {})
            if key in layer_index: raise ValueError(f'duplicate tensor {name}')
            a, b = meta['data_offsets']
            dtype = DTYPES[meta['dtype']]
            shape = tuple(meta['shape'])
            if b-a != int(np.prod(shape))*np.dtype(dtype).itemsize:
                raise ValueError(f'bad source size {name}')
            if a < 0 or b < a or 8+n+b > path.stat().st_size:
                raise ValueError(f'bad source offset {name}')
            layer_index[key] = (str(path), 8+n+a, dtype, shape)
    return result


def layout_spec(bits, hidden, intermediate):
    tiers = [(k, [e for e,b in enumerate(bits) if b == k]) for k in sorted(set(bits))]
    if len(tiers) != 2 or [k for k,_ in tiers] != [3,4]:
        raise ValueError('this cache supports this checkpoint K3/K4 only')
    n = len(bits)
    specs = {}
    for k, ids in tiers:
        specs[f'w13_k{k}'] = ('<i2', [2,len(ids),hidden//16,intermediate//16,16*k])
        specs[f'w2_k{k}'] = ('<i2', [len(ids),intermediate//16,hidden//16,16*k])
    specs.update(gate_suh=('<f2',[n,hidden]), up_suh=('<f2',[n,hidden]),
                 intermediate=('<f2',[n,3*intermediate]), down_svh=('<f2',[n,hidden]),
                 global_to_combined=('<i4',[n]), descriptor_map=('<i4',[3*n]))
    offset = 0; entries = {}
    for name,(dtype,shape) in specs.items():
        offset = (offset+255)//256*256
        size = int(np.prod(shape))*np.dtype(dtype).itemsize
        entries[name] = dict(dtype=dtype, shape=shape, offset=offset, nbytes=size)
        offset += size
    return tiers, entries, offset


def validate(directory, *, key, rank, layer, bits, hidden, intermediate, checksum=True):
    """Return manifest only for a fully published matching artifact; otherwise miss."""
    try:
        directory = Path(directory)
        meta = json.loads((directory/'layout.json').read_text())
        tiers, tensors, size = layout_spec(bits,hidden,intermediate)
        expected = dict(schema=SCHEMA,key=key,rank=rank,layer=layer,bits=list(bits),
                        hidden=hidden,intermediate=intermediate,tensors=tensors,nbytes=size)
        if any(meta.get(k) != v for k,v in expected.items()): return None
        data = directory/'tensors.bin'
        if data.stat().st_size != size: return None
        if checksum and digest(data) != meta['sha256']: return None
        return meta
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_layer(directory, *, key, rank, layer, bits, hidden, intermediate, source):
    directory = Path(directory)
    if validate(directory,key=key,rank=rank,layer=layer,bits=bits,
                hidden=hidden,intermediate=intermediate):
        return dict(layer=layer,status='hit')
    directory.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.layer{layer}-',dir=directory.parent))
    maps = {}; arrays = {}
    try:
        tiers, entries, size = layout_spec(bits,hidden,intermediate)
        with (temp/'tensors.bin').open('wb') as f: f.truncate(size)
        for name,e in entries.items():
            arrays[name] = np.memmap(temp/'tensors.bin',mode='r+',dtype=e['dtype'],
                                     offset=e['offset'],shape=tuple(e['shape']))
        def src(expert,projection,field,shape):
            path,offset,dtype,actual = source[expert,projection,field]
            if tuple(actual) != tuple(shape) or dtype != ('<i4' if field=='mcg' else '<i2' if field=='trellis' else '<f2'):
                raise ValueError(f'source geometry/dtype mismatch {expert} {projection} {field}')
            if path not in maps:
                with open(path,'rb') as f: maps[path]=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
            return np.ndarray(actual,dtype=dtype,buffer=maps[path],offset=offset)
        combined=0
        order=[]
        for tier,(k,ids) in enumerate(tiers):
            for local,expert in enumerate(ids):
                order.append(expert)
                for proj in ('gate_proj','up_proj','down_proj'):
                    # Accept sentinel [] or [1], but validate exact MCG value.
                    marker = source[expert,proj,'mcg']
                    if marker[3] not in ((),(1,)): raise ValueError('bad MCG scalar shape')
                    if int(src(expert,proj,'mcg',marker[3]).reshape(-1)[0]) & 0xffffffff != 0xCBAC1FED:
                        raise ValueError('invalid MCG codebook')
                    h,i = (intermediate,hidden) if proj=='down_proj' else (hidden,intermediate)
                    packed = src(expert,proj,'trellis',(h//16,i//16,k*16))
                    if proj=='down_proj': arrays[f'w2_k{k}'][local]=packed
                    else: arrays[f'w13_k{k}'][0 if proj=='gate_proj' else 1,local]=packed
                arrays['gate_suh'][combined]=src(expert,'gate_proj','suh',(hidden,))
                arrays['up_suh'][combined]=src(expert,'up_proj','suh',(hidden,))
                arrays['down_svh'][combined]=src(expert,'down_proj','svh',(hidden,))
                for block,(proj,field) in enumerate((('gate_proj','svh'),('up_proj','svh'),('down_proj','suh'))):
                    arrays['intermediate'][combined,block*intermediate:(block+1)*intermediate]=src(expert,proj,field,(intermediate,))
                arrays['global_to_combined'][expert]=combined
                arrays['descriptor_map'][combined::len(bits)]=(tier<<8)|local
                combined+=1
        for a in arrays.values(): a.flush()
        del a, packed
        arrays.clear()
        for m in maps.values(): m.close()
        maps.clear()
        meta=dict(schema=SCHEMA,key=key,rank=rank,layer=layer,bits=list(bits),hidden=hidden,
                  intermediate=intermediate,tensors=entries,nbytes=size,sha256=digest(temp/'tensors.bin'))
        with (temp/'layout.json').open('w') as f:
            json.dump(meta,f,sort_keys=True); f.flush();os.fsync(f.fileno())
        with (temp/'tensors.bin').open('rb') as f: os.fsync(f.fileno())
        # Per-rank build lock held by CLI; invalid prior artifact is not reusable.
        if directory.exists(): shutil.rmtree(directory)
        os.replace(temp,directory)
        return dict(layer=layer,status='built',nbytes=size)
    finally:
        arrays.clear()
        for m in maps.values():
            try: m.close()
            except BufferError: pass
        if temp.exists(): shutil.rmtree(temp)


def build_rank(model, cache, rank, loader_version, workers=2, layers=None):
    if not 1 <= workers <= 8: raise ValueError('prep workers must be 1..8')
    key, ident = identity(model,loader_version)
    cfg=json.loads((Path(model)/'config.json').read_text())
    hidden=int(cfg['hidden_size']);intermediate=int(cfg['moe_intermediate_size'])//4
    bitmap=json.loads((Path(model)/'tier_bitmap.json').read_text())
    indices=index_rank(model,rank)
    selected=sorted(indices) if layers is None else layers
    def job(layer):
        return build_layer(Path(cache)/key/f'rank{rank}'/f'layer{layer}',key=key,rank=rank,
                           layer=layer,bits=bitmap[str(layer)]['k'],hidden=hidden,
                           intermediate=intermediate,source=indices[layer])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(job,selected):
            print(json.dumps(result),flush=True)
    return key
