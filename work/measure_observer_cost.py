import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from audit_updates import load,build,native,canonical,root
import hnsw_observer

p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['sift','t2i'],required=True); p.add_argument('--scale',type=int,default=100000); args=p.parse_args()
folder=root/'results/observer_cost'/f'{args.dataset}_{args.scale}'; folder.mkdir(parents=True,exist_ok=True)
if (folder/'summary.json').exists(): raise RuntimeError('Completed microbenchmark exists')
base,allq,metric,dim=load(args.dataset,args.scale); order=sorted(range(len(allq)),key=lambda i:hashlib.sha256(str(i).encode()).hexdigest()); q=np.asarray(allq[np.array(order[:256])],np.float32)
path=build(args.dataset,args.scale,20261030,base,metric,dim); x=hnsw_observer.Index(str(path),dim,metric,args.scale+2000,128); original=native(path,metric,dim,args.scale,128)
for vector in q: x.query(vector,10,False); original.knn_query(vector,k=10,num_threads=1)
rng=np.random.default_rng(20261030); samples=np.zeros((5,len(q),4)); disagreement=0
for repeat in range(5):
    for qi in rng.permutation(len(q)):
        modes=['original','unobserved','observed']; rng.shuffle(modes); outputs={}
        for mode in modes:
            start=time.perf_counter()
            if mode=='original': i,d=original.knn_query(q[qi],k=10,num_threads=1); v=(i[0],d[0])
            else: v=x.query(q[qi],10,mode=='observed')
            samples[repeat,qi,['original','unobserved','observed'].index(mode)]=time.perf_counter()-start; outputs[mode]=v
        for mode in ['unobserved','observed']:
            ai,ad=canonical(outputs['original'][0],outputs['original'][1]); bi,bd=canonical(outputs[mode][0],outputs[mode][1]); disagreement+=int(not np.array_equal(ai,bi) or not np.array_equal(ad,bd))
        start=time.perf_counter(); ok=x.valid(*outputs['observed'][2:]); samples[repeat,qi,3]=time.perf_counter()-start; assert ok
means=samples.mean(axis=0); differences=(means[:,2]-means[:,1])*1e6; bootstrap=np.median(differences[rng.integers(0,len(q),(20000,len(q)))],axis=1)
row={'dataset':args.dataset,'scale':args.scale,'development_queries':len(q),'randomized_repeats':5,'original_us':float(np.median(samples[:,:,0])*1e6),'unobserved_us':float(np.median(samples[:,:,1])*1e6),'observed_us':float(np.median(samples[:,:,2])*1e6),'validation_us':float(np.median(samples[:,:,3])*1e6),'paired_mean_query_recording_increment_us':float(np.median(differences)),'paired_increment_ci95_us':np.quantile(bootstrap,[.025,.975]).tolist(),'native_id_and_distance_disagreements':disagreement,'interpretation':'Warm randomized API calls on development IDs. The first preliminary footprint probe always recorded before unobserved search and is not used to estimate instrumentation cost.'}
assert disagreement==0
np.savez_compressed(folder/'measurements.npz',query_ids=np.array(order[:256]),seconds=samples)
(folder/'summary.json').write_text(json.dumps(row,indent=2)); print(json.dumps(row,indent=2))
