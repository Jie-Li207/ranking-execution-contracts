import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,sys,time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/deps'))
if os.environ.get('ICDE_DEPS'): sys.path.insert(0,os.environ['ICDE_DEPS'])
import h5py
import faiss
threadpool_limits(1)
faiss.omp_set_num_threads(1)
faiss.cvar.distance_compute_blas_threshold=1

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''): h.update(block)
    return h.hexdigest()

def load(scale):
    f=h5py.File(ROOT/'data/sift-128-euclidean.hdf5','r')
    base=f['train']
    queries=np.asarray(f['test'][:],dtype=np.float32)
    truth=np.asarray(f['neighbors'][:,:100],dtype=np.int32)
    return base,queries,truth

def build(backend,base,scale):
    folder=ROOT/'data/indices'
    folder.mkdir(exist_ok=True)
    path=folder/f'{backend}_{scale}_seed20261030.index'
    receipt=path.with_suffix('.json')
    started=time.perf_counter()
    if backend in ['faiss','faiss_sq8']:
        version=faiss.__version__
        if path.exists(): index=faiss.read_index(str(path))
        else:
            index=faiss.IndexHNSWFlat(base.shape[1],16) if backend=='faiss' else faiss.IndexHNSWSQ(base.shape[1],faiss.ScalarQuantizer.QT_8bit_direct,16)
            if backend=='faiss_sq8':
                assert index.is_trained
            index.hnsw.efConstruction=200
            index.hnsw.rng=faiss.RandomGenerator(20261030)
            for start in range(0,scale,10000):
                block=np.asarray(base[start:min(start+10000,scale)],dtype=np.float32)
                if backend=='faiss_sq8': assert np.all((block>=0)&(block<=255)&(block==np.floor(block)))
                index.add(block)
                print('build',backend,start+len(block),scale,round(time.perf_counter()-started,1),flush=True)
            faiss.write_index(index,str(path))
    else:
        import hnswlib
        version=importlib.metadata.version('hnswlib')
        index=hnswlib.Index(space='l2',dim=base.shape[1])
        if path.exists(): index.load_index(str(path),max_elements=scale+1024)
        else:
            index.init_index(max_elements=scale+1024,ef_construction=200,M=16,random_seed=20261030)
            index.set_num_threads(1)
            for start in range(0,scale,10000):
                block=np.asarray(base[start:min(start+10000,scale)],dtype=np.float32)
                index.add_items(block,np.arange(start,start+len(block)),num_threads=1)
                print('build',backend,start+len(block),scale,round(time.perf_counter()-started,1),flush=True)
            index.save_index(str(path))
        index.set_num_threads(1)
    if not receipt.exists():
        receipt.write_text(json.dumps({'backend':backend,'version':version,'scale':scale,'seconds':time.perf_counter()-started,'index_bytes':path.stat().st_size,'index_sha256':digest(path),'M':16,'efConstruction':200,'seed':20261030,'threads':1},indent=2))
    return index,json.loads(receipt.read_text())

def search(index,backend,q,k,ef):
    if backend in ['faiss','faiss_sq8']:
        index.hnsw.efSearch=ef
        d,i=index.search(np.ascontiguousarray(q),k)
    else:
        index.set_ef(ef)
        i,d=index.knn_query(np.ascontiguousarray(q),k=k,num_threads=1)
    return np.asarray(d),np.asarray(i,dtype=np.int32)

def compare(da,ia,db,ib,k):
    changed=ia[:,:k]!=ib[:,:k]
    strict=changed&(np.abs(da[:,:k].astype(np.float64)-db[:,:k].astype(np.float64))>1e-6)
    return {'ordered_prefix_changes':int(np.any(changed,axis=1).sum()),'first_changes':int(changed[:,0].sum()),'strict_first_changes':int(strict[:,0].sum()),'strict_prefix_changes':int(np.any(strict,axis=1).sum()),'queries':len(ia)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['faiss','faiss_sq8','hnswlib'],required=True)
    p.add_argument('--scale',type=int,required=True)
    p.add_argument('--phase',choices=['development','confirmation'],required=True)
    args=p.parse_args()
    out=ROOT/'results/native_ann'/f'{args.backend}_{args.scale}_{args.phase}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise RuntimeError('Completed result directory exists; do not overwrite it')
    base,queries,truth=load(args.scale)
    ordered=sorted(range(len(queries)),key=lambda i:hashlib.sha256(str(i).encode()).hexdigest())
    take=np.array(ordered[:256] if args.phase=='development' else ordered[256:2304])
    q=queries[take]
    index,build_receipt=build(args.backend,base,args.scale)
    if args.scale==len(base) and args.scale==1000000: exact=truth[take]
    else:
        flat=faiss.IndexFlatL2(base.shape[1])
        for start in range(0,args.scale,10000): flat.add(np.asarray(base[start:min(start+10000,args.scale)],dtype=np.float32))
        _,exact=flat.search(q,100)
        del flat
    records=[]
    arrays={'query_ids':take,'ground_truth':exact}
    ks=[1,5,10,20,50,100]
    efs=[16,64,128]
    for ef in efs:
        outputs={}
        for k in ks:
            samples=[]
            for repeat in range(3):
                tick=time.perf_counter()
                d,i=search(index,args.backend,q,k,ef)
                samples.append(time.perf_counter()-tick)
                if repeat:
                    assert np.array_equal(i,previous_i) and np.array_equal(d,previous_d)
                previous_i,previous_d=i.copy(),d.copy()
            outputs[k]=(d,i)
            arrays[f'ef{ef}_k{k}_ids']=i
            arrays[f'ef{ef}_k{k}_distances']=d
            recalls=np.array([len(set(a)&set(b[:k]))/k for a,b in zip(i,exact)])
            rec={'ef':ef,'k':k,'recall':float(recalls.mean()),'repeat_seconds':samples}
            records.append(rec)
            print('search',args.backend,args.scale,args.phase,rec,flush=True)
        for rec in records:
            if rec['ef']==ef: rec['versus_k100']=compare(*outputs[rec['k']],*outputs[100],rec['k'])
        d0,i0=outputs[10]
        dr,ir=search(index,args.backend,q[::-1],10,ef)
        reverse=compare(d0,i0,dr[::-1],ir[::-1],10)
        dp,ip=[],[]
        for start in range(0,len(q),64):
            d,i=search(index,args.backend,q[start:start+64],10,ef)
            dp.append(d); ip.append(i)
        partition=compare(d0,i0,np.vstack(dp),np.vstack(ip),10)
        arrays[f'ef{ef}_reverse_ids']=ir[::-1]
        arrays[f'ef{ef}_partition_ids']=np.vstack(ip)
        records.append({'ef':ef,'control':'same_query_batch','reverse':reverse,'partition64':partition})
        (out/'partial.json').write_text(json.dumps(records,indent=2))
    rng=np.random.default_rng(20261030)
    for rec in records:
        if 'k' not in rec: continue
        ef,k=rec['ef'],rec['k']
        changed=np.any(arrays[f'ef{ef}_k{k}_ids']!=arrays[f'ef{ef}_k100_ids'][:,:k],axis=1)
        means=[rng.choice(changed,size=len(changed),replace=True).mean() for _ in range(2000)]
        rec['prefix_change_fraction_CI95']=np.quantile(means,[.025,.975]).tolist()
    np.savez_compressed(out/'outputs.npz',**arrays)
    summary={'backend':args.backend,'scale':args.scale,'phase':args.phase,'query_count':len(take),'query_selection':'SHA256 ordered official query IDs; development first256, confirmation next2048','protocol_sha256':digest(ROOT/'results/revision_protocol.json'),'build':build_receipt,'records':records,'scope':'Native released ANN implementation on the complete selected gallery. An ANN API does not promise equality across k; strict prefix preservation is a separately declared requirement. Equal-distance changes are counted separately. Query labels do not tune any configuration.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('DONE',args.backend,args.scale,args.phase,flush=True)

if __name__=='__main__': main()
