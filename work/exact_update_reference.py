import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from audit_updates import load,root

def topk(d,ids,k=10):
    partition=np.argpartition(d,k-1,axis=1)[:,:k]
    cut=np.take_along_axis(d,partition,axis=1).max(axis=1)
    out_i=np.zeros((len(d),k),np.int64); out_d=np.zeros((len(d),k))
    for row in range(len(d)):
        candidates=np.flatnonzero(d[row]<=cut[row])
        order=np.lexsort((ids[row,candidates],d[row,candidates]))[:k]
        chosen=candidates[order]; out_i[row]=ids[row,chosen]; out_d[row]=d[row,chosen]
    return out_i,out_d
def distance(q,x,metric):
    if metric=='ip': return 1-q@x.T
    return (q*q).sum(axis=1)[:,None]+(x*x).sum(axis=1)[None,:]-2*q@x.T
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['sift','t2i'],required=True); p.add_argument('--scale',type=int,default=100000); args=p.parse_args()
    folder=root/'results/update_exact'/f'{args.dataset}_{args.scale}'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists(): raise RuntimeError('Completed exact reference exists')
    base,allq,metric,dim=load(args.dataset,args.scale)
    files=sorted((root/'results/update_confirmation').glob(f'{args.dataset}_{args.scale}_*/summary.json'))
    if not files: raise RuntimeError('No mechanism results')
    take=np.load(files[0].parent/'before_ef128.npz')['query_ids']; q=np.asarray(allq[take],float)
    before=folder/'exact_before.npz'
    if before.exists(): z=np.load(before); truth_i=z['ids']; truth_d=z['distances']
    else:
        truth_i=np.full((len(q),10),-1,np.int64); truth_d=np.full((len(q),10),np.inf); t=time.perf_counter()
        for start in range(0,args.scale,10000):
            x=np.asarray(base[start:min(start+10000,args.scale)],float); assert np.isfinite(x).all()
            for begin in range(0,len(q),64):
                block=q[begin:begin+64]; d=distance(block,x,metric)
                bi,bd=topk(d,np.broadcast_to(np.arange(start,start+len(x)),d.shape))
                merged_i=np.concatenate([truth_i[begin:begin+len(block)],bi],axis=1); merged_d=np.concatenate([truth_d[begin:begin+len(block)],bd],axis=1)
                ii,dd=topk(merged_d,merged_i); truth_i[begin:begin+len(block)]=ii; truth_d[begin:begin+len(block)]=dd
            if start%100000==0: print('Exact scan',args.dataset,args.scale,start+len(x),round(time.perf_counter()-t,1),flush=True)
        np.savez_compressed(before,query_ids=take,ids=truth_i,distances=truth_d)
    extra=np.asarray(base[args.scale:args.scale+1024],float); dist=distance(q,extra,metric)
    after={}
    for count in [1,16,256,1024]:
        ii,dd=topk(np.concatenate([truth_d,dist[:,:count]],axis=1),np.concatenate([truth_i,np.broadcast_to(np.arange(args.scale,args.scale+count),(len(q),count))],axis=1))
        after[count]=(ii,dd); np.savez_compressed(folder/f'exact_after_{count}.npz',query_ids=take,ids=ii,distances=dd)
    rows=[]
    for file in files:
        for row in json.loads(file.read_text()):
            z=np.load(file.parent/f'after_ef{row["ef"]}_append{row["append"]}.npz'); old=np.load(file.parent/f'before_ef{row["ef"]}.npz')
            true_after=after[row['append']][0]; exact_changed=np.any(truth_i!=true_after,axis=1)
            false_far=exact_changed & z['far']; assert not false_far.any()
            recall_before=np.any(old['ids'][:,:,None]==truth_i[:,None,:],axis=2).sum(axis=1)/10
            recall_after=np.any(z['ids'][:,:,None]==true_after[:,None,:],axis=2).sum(axis=1)/10
            rows.append({'dataset':args.dataset,'scale':args.scale,'seed':row['seed'],'ef':row['ef'],'append':row['append'],'queries':len(q),'native_recall_before':float(recall_before.mean()),'native_recall_after':float(recall_after.mean()),'far_queries_with_exact_change':int(false_far.sum()),'far_native_witnesses_verified':row['far_changes']})
    (folder/'summary.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
