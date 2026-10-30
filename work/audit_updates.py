import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
import hashlib,json,sys,time,argparse
import numpy as np
root=Path(os.environ.get('ICDE_UPDATE_ROOT',Path(__file__).resolve().parents[1])); previous=Path(os.environ.get('ICDE_NATIVE_ROOT',root))
sys.path[:0]=[str(root/'work/native_observer'),str(previous/'work/deps')]
import hnswlib,h5py,hnsw_observer

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()
def canonical(ids,d):
    o=np.lexsort((ids,d),axis=-1)
    return np.take_along_axis(ids,o,axis=-1),np.take_along_axis(d,o,axis=-1)
def native(path,metric,dim,scale,ef):
    x=hnswlib.Index(space=metric,dim=dim); x.load_index(str(path),max_elements=scale+2000); x.set_num_threads(1); x.set_ef(ef); return x
def load(dataset,scale):
    if dataset=='sift':
        f=h5py.File(previous/'data/sift-128-euclidean.hdf5','r')
        return f['train'],np.asarray(f['test'][:],np.float32),'l2',128
    return np.memmap(root/('data/t2i_base_1m.fbin' if scale==1000000 else 'data/t2i_base_100k.fbin'),dtype='<f4',offset=8,shape=(scale+1024,200),mode='r'),np.memmap(root/'data/t2i_queries.fbin',dtype='<f4',offset=8,shape=(100000,200),mode='r'),'ip',200
def build(dataset,scale,seed,base,metric,dim):
    if dataset=='sift' and scale==100000 and seed==20261030: return previous/'data/indices/hnswlib_100000_seed20261030.index'
    if dataset=='t2i' and scale==100000 and seed==20261030 and (root/'data/t2i_hnsw_100k.index').exists(): return root/'data/t2i_hnsw_100k.index'
    path=root/'data'/f'{dataset}_{scale}_{seed}.index'
    if dataset=='t2i' and scale==100000 and seed==20261030: path=root/'data/t2i_hnsw_100k.index'
    if path.exists() and path.with_suffix('.json').exists(): return path
    t=time.perf_counter(); x=hnswlib.Index(space=metric,dim=dim)
    x.init_index(max_elements=scale+2000,ef_construction=200,M=16,random_seed=seed); x.set_num_threads(1)
    for start in range(0,scale,10000):
        b=np.asarray(base[start:min(start+10000,scale)],np.float32)
        x.add_items(b,np.arange(start,start+len(b)),num_threads=1)
        if start%100000==0: print('build',dataset,scale,seed,start+len(b),round(time.perf_counter()-t,1),flush=True)
    x.save_index(str(path)); path.with_suffix('.json').write_text(json.dumps({'seconds':time.perf_counter()-t,'sha256':digest(path),'bytes':path.stat().st_size,'seed':seed,'M':16,'ef_construction':200,'threads':1},indent=2)); return path
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['sift','t2i'],required=True); p.add_argument('--scale',type=int,default=100000); p.add_argument('--seed',type=int,default=20261030); args=p.parse_args()
    out=root/'results/update_confirmation'/f'{args.dataset}_{args.scale}_{args.seed}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise RuntimeError('Completed results exist')
    base,queries,metric,dim=load(args.dataset,args.scale)
    order=sorted(range(len(queries)),key=lambda i:hashlib.sha256(str(i).encode()).hexdigest())
    take=np.array(order[2304:4352] if args.dataset=='sift' else order[256:2304])
    q=np.asarray(queries[take],np.float32); extra=np.asarray(base[args.scale:args.scale+1024],np.float32)
    path=build(args.dataset,args.scale,args.seed,base,metric,dim)
    rows=[]
    for ef in [32,128]:
        clean=native(path,metric,dim,args.scale,ef)
        old_i,old_d=canonical(*clean.knn_query(q,k=10,num_threads=1)); del clean
        observed=hnsw_observer.Index(str(path),dim,metric,args.scale+2000,ef)
        certs=[observed.query(v,10,True) for v in q]
        oi,od=canonical(np.array([c[0] for c in certs]),np.array([c[1] for c in certs])); assert np.array_equal(oi,old_i)
        footprint_sizes=np.array([len(c[2]) for c in certs]); offsets=np.r_[0,np.cumsum(footprint_sizes)]
        old_exact_cut=[]
        for qi,ids in enumerate(old_i):
            v=np.asarray(base[np.sort(ids)],np.float64) if args.dataset=='sift' else np.asarray(base[ids],np.float64)
            d=((v-q[qi].astype(float))**2).sum(axis=1) if metric=='l2' else 1-v@q[qi].astype(float)
            old_exact_cut.append(d.max())
        old_exact_cut=np.array(old_exact_cut)
        np.savez_compressed(out/f'before_ef{ef}.npz',query_ids=take,ids=old_i,distances=old_d,cutoff_float64=old_exact_cut,footprint_offsets=offsets,footprint_nodes=np.concatenate([c[2] for c in certs]))
        del observed
        for count in [1,16,256,1024]:
            x=hnsw_observer.Index(str(path),dim,metric,args.scale+2000,ef)
            fresh=native(path,metric,dim,args.scale,ef)
            t=time.perf_counter(); x.append(extra[:count],np.arange(args.scale,args.scale+count,dtype=np.uint64)); observed_insert=time.perf_counter()-t
            t=time.perf_counter(); fresh.add_items(extra[:count],np.arange(args.scale,args.scale+count),num_threads=1); native_insert=time.perf_counter()-t
            ni,nd=canonical(*fresh.knn_query(q,k=10,num_threads=1))
            for repeat in range(2):
                ri,rd=canonical(*fresh.knn_query(q,k=10,num_threads=1)); assert np.array_equal(ni,ri) and np.array_equal(nd,rd)
            vals=np.array([x.valid(*c[2:]) for c in certs]); outputs=[x.query(v,10,False) for v in q]
            xi,xd=canonical(np.array([c[0] for c in outputs]),np.array([c[1] for c in outputs])); assert np.array_equal(xi,ni) and np.array_equal(xd,nd)
            distance=[]
            for begin in range(0,len(q),64):
                qq=q[begin:begin+64].astype(float); xx=extra[:count].astype(float)
                d=(qq*qq).sum(axis=1)[:,None]+(xx*xx).sum(axis=1)[None,:]-2*qq@xx.T if metric=='l2' else 1-qq@xx.T
                distance.extend(d.min(axis=1))
            tolerance=1e-3 if metric=='l2' else 1e-5
            far=np.array(distance)>old_exact_cut+tolerance
            changed=np.any(old_i!=ni,axis=1); strict=np.any((old_i!=ni)&(abs(old_d.astype(float)-nd)>tolerance),axis=1)
            false=vals&changed; assert not false.any()
            same=None
            if count==1024 and ef==128:
                a=out/'observed.index'; b=out/'unmodified.index'; x.save(str(a)); fresh.save_index(str(b)); same=digest(a)==digest(b); assert same; a.unlink(); b.unlink()
            row={'dataset':args.dataset,'scale':args.scale,'seed':args.seed,'ef':ef,'append':count,'queries':len(q),'far_queries':int(far.sum()),'changed':int(changed.sum()),'far_changes':int((far&changed).sum()),'far_strict_changes':int((far&strict).sum()),'far_first_changes':int((far&(old_i[:,0]!=ni[:,0])).sum()),'changes_only_old_ids':int((changed&np.all(ni<args.scale,axis=1)).sum()),'admitted':int(vals.sum()),'false_admitted':int(false.sum()),'conservative_rejections':int((~vals&~changed).sum()),'median_nodes':float(np.median(footprint_sizes)),'median_certificate_bytes':float(np.median(12*footprint_sizes+16)),'version_bytes':x.version_bytes(),'observed_insert_seconds':observed_insert,'native_insert_seconds':native_insert,'serialized_native_equal':same,'far_witness_ids':take[far&changed].tolist(),'native_id_and_distance_errors':0}
            rows.append(row); print(row,flush=True)
            np.savez_compressed(out/f'after_ef{ef}_append{count}.npz',query_ids=take,ids=ni,distances=nd,far=far,min_inserted_distance=distance,admitted=vals)
            del x,fresh
        (out/'partial.json').write_text(json.dumps(rows,indent=2))
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__': main()
