#!/usr/bin/env python3
"""Streaming, byte-preserving fragment export; stdlib only, no GPU/torch.

Non-expert weights remain full (standard vLLM loader shards them). Original
expert tensor names and all bytes are retained, but only owned fragments are
copied. Headers/indexes are rebuilt; source files are never modified.
"""
import argparse,hashlib,json,os,shutil,struct,time
from pathlib import Path
from fragments import PATTERN,owner,placement

def header(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if n>256<<20:raise ValueError('oversized header')
        return json.loads(f.read(n)),8+n

def export_file(src,dst,rank):
    h,base=header(src)
    selected={};offset=0
    for name,m in h.items():
        if name=='__metadata__':continue
        match=PATTERN.fullmatch(name)
        if match and owner(int(match[3]),int(match[5]),int(match[2]))!=rank:continue
        start,end=m['data_offsets'];size=end-start
        assert size>=0 and base+end<=src.stat().st_size,(src,name)
        selected[name]={**m,'data_offsets':[offset,offset+size]};offset+=size
    out={'__metadata__':{'format':'pt','glm6_layout':'exact-fragments-v1','rank':str(rank)},**selected}
    data=json.dumps(out,separators=(',',':')).encode();data+=b' '*((-len(data))%8)
    expected=8+len(data)+offset
    manifest=dst.with_suffix('.sha256')
    if dst.exists() and manifest.exists() and dst.stat().st_size==expected:
        return selected,offset,'existing'
    temp=dst.with_suffix('.tmp');sha=hashlib.sha256()
    with src.open('rb') as inp,temp.open('wb') as f:
        for b in (struct.pack('<Q',len(data)),data):f.write(b);sha.update(b)
        for name,m in selected.items():
            start,end=h[name]['data_offsets'];inp.seek(base+start);left=end-start
            while left:
                b=inp.read(min(left,8<<20))
                if not b:raise EOFError((src,name,left))
                f.write(b);sha.update(b);left-=len(b)
        f.flush();os.fsync(f.fileno())
    assert temp.stat().st_size==expected
    os.replace(temp,dst);manifest.write_text(sha.hexdigest()+'  '+dst.name+'\n')
    return selected,offset,'copied'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--rank',type=int,required=True);ap.add_argument('--layers',type=int,nargs='*')
    a=ap.parse_args();assert 0<=a.rank<6
    src=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    sourceconfig=json.loads((src/'config.json').read_text());assert sourceconfig['num_attention_heads']==64
    assert sourceconfig['moe_intermediate_size']==2048 and sourceconfig['n_routed_experts']==256
    manifestsha=hashlib.sha256((src/'MANIFEST.sha256').read_bytes()).hexdigest()
    identity=dict(schema='exact-fragments-v1',rank=a.rank,source_manifest_sha256=manifestsha,
                  placement='(4*expert + source_part + layer) % 6',source_tp=4,fragment_width=512,
                  logical_attention_heads=64,physical_attention_heads=72,logical_vocab_size=154880,
                  logical_experts=256,logical_expert_width=2048)
    if (out/'glm6_layout.json').exists():assert json.loads((out/'glm6_layout.json').read_text())==identity
    (out/'glm6_layout.json').write_text(json.dumps(identity,indent=2)+'\n')
    for file in src.iterdir():
        if file.is_file() and file.suffix!='.safetensors' and file.name not in ('config.json','model.safetensors.index.json','MANIFEST.sha256'):
            shutil.copy2(file,out/file.name)
    shutil.copy2(src/'config.json',out/'source.config.json')
    shutil.copy2(src/'MANIFEST.sha256',out/'SOURCE.MANIFEST.sha256')
    config={**sourceconfig,'num_attention_heads':72,'glm6_layout':identity}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    weightmap={};total=0
    for file in sorted(src.glob('*.safetensors')):
        if a.layers is not None and file.name not in [f'model-layer-{l:03d}.safetensors' for l in a.layers]:continue
        t=time.monotonic();selected,size,status=export_file(file,out/file.name,a.rank)
        weightmap.update({n:file.name for n in selected});total+=size
        print(json.dumps(dict(rank=a.rank,file=file.name,status=status,bytes=size,seconds=round(time.monotonic()-t,3))),flush=True)
    index=dict(metadata={'total_size':total},weight_map=weightmap)
    (out/'model.safetensors.index.json').write_text(json.dumps(index,sort_keys=True)+'\n')
    (out/'MANIFEST.sha256').write_text(''.join(p.read_text() for p in sorted(out.glob('*.sha256')) if p.name not in ('SOURCE.MANIFEST.sha256','MANIFEST.sha256')))
    (out/'EXPORT_COMPLETE').write_text(json.dumps(dict(rank=a.rank,payload_bytes=total,tensors=len(weightmap)))+'\n')
    print('EXPORT_COMPLETE',a.rank,total,flush=True)
if __name__=='__main__':main()
