import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
import argparse,json,sys
import numpy as np
from audit_updates import load,build,native,canonical,root

p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['sift','t2i'],required=True); p.add_argument('--scale',type=int,default=100000); p.add_argument('--phase',choices=['development','confirmation','timing_followup'],default='development'); p.add_argument('--wrapper',choices=['native','matched'],default='native'); args=p.parse_args()
out=root/'results'/('service_'+args.phase+('_matched' if args.wrapper=='matched' else ''))/f'{args.dataset}_{args.scale}'
rows=json.loads((out/'summary.json').read_text()); take=np.load(out/'query_ids.npy')
base,allq,metric,dim=load(args.dataset,args.scale); queries=np.asarray(allq[take],np.float32); extra=np.asarray(base[args.scale:args.scale+1024],np.float32); path=build(args.dataset,args.scale,20261030,base,metric,dim)
references={}
for update in sorted({row['config']['update'] for row in rows}):
    reference=out/f'oracle_update{update}.npz'
    if reference.exists(): z=np.load(reference); references[update]=(z['ids'],z['distances']); continue
    idx=native(path,metric,dim,args.scale,128); ids=[]; distances=[]
    epochs=max(len(row['updates']) for row in rows if row['config']['update']==update)
    for epoch in range(epochs+1):
        if epoch and update:
            begin=(epoch-1)*update; idx.add_items(extra[begin:begin+update],np.arange(args.scale+begin,args.scale+begin+update),num_threads=1)
        if epoch and not update: ni,nd=ids[0],distances[0]
        else: ni,nd=canonical(*idx.knn_query(queries,k=10,num_threads=1))
        ids.append(ni); distances.append(nd)
    references[update]=(np.array(ids),np.array(distances))
    np.savez_compressed(reference,ids=ids,distances=distances,query_ids=take)
    print('Fresh oracle',args.dataset,update,epochs+1,len(queries),flush=True); del idx
results=[]
for row in rows:
    z=np.load(out/(row['label']+'.npz')); update=row['config']['update']; ni,nd=references[update]; expected=ni[z['epochs'],z['trace']]; expected_d=nd[z['epochs'],z['trace']]
    ids=z['ids']; epoch=z['epochs']; values=np.zeros_like(ids,dtype=np.int32)
    if update:
        mask=ids>=args.scale; values[mask]=(ids[mask]-args.scale)//update+1
    for e in range(1,int(epoch.max())+1):
        low=(e*7919)%max(1,args.scale-256)
        values[(epoch[:,None]>=e)&(ids>=low)&(ids<low+256)]=e
    bad_i=np.any(ids!=expected,axis=1); bad_d=np.any(z['distances']!=expected_d,axis=1); bad_v=np.any(z['values']!=values,axis=1)
    result={'label':row['label'],'requests':len(ids),'wrong_ids':int(bad_i.sum()),'wrong_distances':int(bad_d.sum()),'wrong_values':int(bad_v.sum()),'observed_epochs':sorted(set(epoch.tolist())),'peak_within_budget':row['peak_auxiliary_payload_bytes']<=row['config']['bytes']}
    results.append(result)
(out/'verification.json').write_text(json.dumps(results,indent=2))
bad=[r for r in results if r['wrong_ids'] or r['wrong_values'] or r['wrong_distances'] or not r['peak_within_budget']]
print(json.dumps({'runs':len(results),'requests':sum(r['requests'] for r in results),'failed_runs':bad},indent=2),flush=True)
if bad: raise RuntimeError('Verification failed')
