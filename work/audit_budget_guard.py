from pathlib import Path
import itertools,json,time
import numpy as np
from budget_cache import canonical,prefix_admitted

root=Path(__file__).resolve().parents[1]
rows=[]
for folder in sorted((root/'results/native_ann').glob('hnswlib_*_confirmation')):
    if not (folder/'outputs.npz').exists(): continue
    z=np.load(folder/'outputs.npz')
    for ef in [16,64,128]:
        d,i={},{}
        for k in [1,5,10,20,50,100]:
            pairs=[canonical(a,b) for a,b in zip(z[f'ef{ef}_k{k}_distances'],z[f'ef{ef}_k{k}_ids'])]
            d[k]=np.stack([v[0] for v in pairs])
            i[k]=np.stack([v[1] for v in pairs])
        admitted=rejected=mismatch=unsafe=0
        guard_ns=[]
        pair_rows=[]
        for k,K in itertools.combinations([1,5,10,20,50,100],2):
            count=fail=0
            for qi in range(len(z['query_ids'])):
                t=time.perf_counter_ns()
                ok=prefix_admitted(ef,k,K,d[K][qi])
                guard_ns.append(time.perf_counter_ns()-t)
                changed=not np.array_equal(i[k][qi],i[K][qi,:k])
                unsafe+=int(changed)
                admitted+=int(ok); rejected+=int(not ok)
                mismatch+=int(ok and changed)
                count+=int(ok); fail+=int(ok and changed)
            pair_rows.append({'k':k,'K':K,'admitted':count,'mismatches':fail})
        rows.append({'workload':folder.name,'ef':ef,'comparisons':admitted+rejected,'admitted':admitted,'rejected':rejected,'false_admissions':mismatch,'unconditional_mismatches':unsafe,'guard_p50_ns':float(np.median(guard_ns)),'guard_p95_ns':float(np.quantile(guard_ns,.95)),'pairs':pair_rows})
        print(rows[-1],flush=True)
(root/'results/native_budget_guard.json').write_text(json.dumps({'rows':rows,'scope':'Source-derived sufficient guard for pinned hnswlib searchKnn, fixed graph/metric/configuration, no filters/deletions, canonical distance-then-ID order and strict prefix-cut gap. Cost is isolated Python guard cost; service cost is separate.'},indent=2))
assert all(r['false_admissions']==0 for r in rows)
