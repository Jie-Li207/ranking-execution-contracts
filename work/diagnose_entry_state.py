import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
import json
import numpy as np
from audit_updates import load,build,root
import hnsw_observer

rows=[]
for dataset,scale in [('sift',100000),('t2i',100000),('t2i',1000000)]:
    base,queries,metric,dim=load(dataset,scale)
    for seed in ([20261030] if scale==1000000 else [20261030,20261031,20261032]):
        path=build(dataset,scale,seed,base,metric,dim)
        x=hnsw_observer.Index(str(path),dim,metric,scale+2000,128)
        before=x.query(queries[0],10,False)[-2:]
        start=0
        for count in [1,16,256,1024]:
            x.append(np.asarray(base[scale+start:scale+count],np.float32),np.arange(scale+start,scale+count,dtype=np.uint64))
            after=x.query(queries[0],10,False)[-2:]
            rows.append({'dataset':dataset,'scale':scale,'seed':seed,'append':count,'entry_before':int(before[0]),'level_before':int(before[1]),'entry_after':int(after[0]),'level_after':int(after[1]),'entry_state_changed':before!=after})
            start=count
        del x
(root/'results/entry_state_diagnostic.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
