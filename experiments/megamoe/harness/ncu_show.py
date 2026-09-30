import csv,io,sys
txt=open(sys.argv[1]).read(); i=txt.index('"ID"')
rows=list(csv.DictReader(io.StringIO(txt[i:])))
keep=("Duration","Compute (SM) Throughput","Memory Throughput","DRAM Throughput","Achieved Occupancy","Theoretical Occupancy","Executed Ipc","Registers","Bank","pipe","Tensor","Warp Cycles Per Issued","Eligible Warps","Active Warps","L1/TEX Hit","L2 Hit","Block Limit","Busy","Issue Slots Busy","Local Memory")
seen=set()
for r in rows:
    n=r["Metric Name"]; k=(r["ID"],n)
    if any(t.lower() in n.lower() for t in keep) and k not in seen:
        seen.add(k); print(r["ID"],r["Kernel Name"][:20],"|",r["Section Name"][:20],"|",n[:58],"|",r["Metric Value"],r["Metric Unit"])
