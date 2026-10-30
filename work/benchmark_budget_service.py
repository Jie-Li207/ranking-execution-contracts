import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,sqlite3,sys,threading,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/deps'))
if os.environ.get('ICDE_DEPS'): sys.path.insert(0,os.environ['ICDE_DEPS'])
import hnswlib,h5py
from budget_cache import canonical,prefix_admitted

parser=argparse.ArgumentParser()
parser.add_argument('--cache-bytes',type=int,default=0)
parser.add_argument('--invalid-only',action='store_true')
options=parser.parse_args()
OUT=ROOT/('results/budget_service_bytes'+str(options.cache_bytes) if options.cache_bytes else 'results/budget_service_v2')
if options.invalid_only: OUT=OUT.with_name(OUT.name+'_invalid')
OUT.mkdir(exist_ok=True)
PROTOCOL=json.loads((ROOT/'results/service_protocol.json').read_text())
INDEX=ROOT/'data/indices/hnswlib_100000_seed20261030.index'

def new_index():
    idx=hnswlib.Index(space='l2',dim=128)
    idx.load_index(str(INDEX),max_elements=100256)
    idx.set_num_threads(1)
    return idx

def query(idx,q,k,ef):
    idx.set_ef(int(ef))
    ids,d=idx.knn_query(q.reshape(1,-1),k=int(k),num_threads=1)
    return canonical(d[0],ids[0].astype(np.int32))

def traces(seed,selected):
    rng=np.random.default_rng(seed)
    ids=[]
    for r in range(4096):
        if r and rng.random()<.5: ids.append(ids[int(rng.integers(max(0,r-32),r))])
        else: ids.append(int(rng.choice(selected)))
    return np.column_stack([ids,rng.choice([1,5,10,20,50,100],4096),rng.choice([16,64,128],4096)]).astype(np.int32)

def reference(trace,queries,extra,seed):
    idx=new_index()
    out=np.full((4096,100),-1,np.int32)
    values=np.full_like(out,-1)
    for step,(qi,k,ef) in enumerate(trace):
        epoch=step//1024
        if step==2048: idx.add_items(extra,np.arange(100000,100256),num_threads=1)
        _,ids=query(idx,queries[qi],k,ef)
        out[step,:k]=ids
        values[step,:k]=(ids%8==0)*([0,1,1,2][epoch]-(ids>=100000).astype(np.int32))
    np.savez_compressed(OUT/f'reference_{seed}.npz',trace=trace,ids=out,values=values)
    return out,values

def run(trace,queries,extra,expected,expected_values,seed,clients,policy):
    label=f'{seed}_clients{clients}_{policy}'
    database=OUT/(label+'.sqlite')
    if database.exists(): raise RuntimeError('Do not overwrite a completed or failed workload')
    idx=new_index()
    db=sqlite3.connect(database)
    db.execute('pragma journal_mode=WAL')
    db.execute('pragma synchronous=NORMAL')
    db.execute('create table record(id integer primary key, revision integer not null)')
    db.executemany('insert into record values(?,0)',((i,) for i in range(100000)))
    db.commit()
    cache=OrderedDict()
    cache_lock=threading.Lock()
    search_lock=threading.Lock()
    counters={'hits':0,'prefix_hits':0,'misses':0,'guard_rejects':0,'evictions':0,'native_calls':0,'cache_bytes':0,'peak_cache_bytes':0}
    def entry_size(key,value):
        d,i,v=value
        return d.nbytes+i.nbytes+(0 if v is None else v.nbytes)+len(json.dumps(key))
    latencies=np.zeros(4096)
    stages=np.zeros((4096,3))
    returned=np.full((4096,100),-1,np.int32)
    joined=np.full_like(returned,-1)
    hit_mask=np.zeros(4096,bool)
    cross_mask=np.zeros(4096,bool)
    updates=[]
    peak_payload=0
    start=time.perf_counter()
    for epoch in range(4):
        begin=time.perf_counter()
        if epoch in [1,3]:
            db.execute('update record set revision=revision+1 where id%8=0')
            db.commit()
        if epoch==2:
            idx.add_items(extra,np.arange(100000,100256),num_threads=1)
            db.executemany('insert into record values(?,0)',((i,) for i in range(100000,100256)))
            db.commit()
        updates.append({'epoch':epoch,'kind':'metadata' if epoch in [1,3] else 'index_append' if epoch==2 else 'initial','seconds':time.perf_counter()-begin})
        index_epoch=int(epoch>=2)
        metadata_epoch=[0,1,1,2][epoch]
        def worker(worker_id):
            conn=sqlite3.connect(database)
            conn.execute('pragma query_only=ON')
            for step in range(epoch*1024+worker_id,(epoch+1)*1024,clients):
                qi,k,ef=map(int,trace[step])
                t=time.perf_counter()
                if policy=='coarse_result_exact_k': key=(index_epoch,metadata_epoch,qi,k,ef)
                elif policy=='ids_exact_k': key=(index_epoch,qi,k,ef)
                elif policy=='ids_guarded_budget': key=(index_epoch,qi,ef,max(ef,k))
                elif policy=='invalid_unversioned_prefix': key=(qi,ef)
                else: key=None
                value=None
                with cache_lock:
                    old=cache.get(key) if key is not None else None
                    if old is not None:
                        d0,i0,j0=old
                        ok=prefix_admitted(ef,k,len(i0),d0) if policy=='ids_guarded_budget' else len(i0)>=k
                        if ok:
                            value=d0[:k].copy(),i0[:k].copy(),None if j0 is None else j0[:k].copy()
                            cache.move_to_end(key)
                            counters['hits']+=1
                            counters['prefix_hits']+=int(k<len(i0))
                            hit_mask[step]=True
                            cross_mask[step]=k<len(i0)
                        else: counters['guard_rejects']+=1
                    if value is None: counters['misses']+=1
                checked=time.perf_counter()
                if value is None:
                    with search_lock:
                        d,ids=query(idx,queries[qi],k,ef)
                    with cache_lock: counters['native_calls']+=1
                    cached_values=None
                else: d,ids,cached_values=value
                searched=time.perf_counter()
                if cached_values is None:
                    conn.execute('begin')
                    vals=np.array([conn.execute('select revision from record where id=?',(int(i),)).fetchone()[0] for i in ids],np.int32)
                    conn.commit()
                else: vals=cached_values
                if value is None and key is not None:
                    with cache_lock:
                        old=cache.get(key)
                        if old is None or len(old[1])<=len(ids):
                            if old is not None: counters['cache_bytes']-=entry_size(key,old)
                            cache[key]=(d,ids,vals if policy in ['coarse_result_exact_k','invalid_unversioned_prefix'] else None)
                            counters['cache_bytes']+=entry_size(key,cache[key])
                        cache.move_to_end(key)
                        while len(cache)>256 or (options.cache_bytes and counters['cache_bytes']>options.cache_bytes):
                            removed_key,removed=cache.popitem(last=False)
                            counters['cache_bytes']-=entry_size(removed_key,removed)
                            counters['evictions']+=1
                        counters['peak_cache_bytes']=max(counters['peak_cache_bytes'],counters['cache_bytes'])
                end=time.perf_counter()
                returned[step,:k]=ids
                joined[step,:k]=vals
                latencies[step]=end-t
                stages[step]=[checked-t,searched-checked,end-searched]
            conn.close()
        with ThreadPoolExecutor(max_workers=clients) as pool:
            list(pool.map(worker,range(clients)))
        payload=sum(d.nbytes+i.nbytes+(0 if v is None else v.nbytes)+len(json.dumps(key)) for key,(d,i,v) in cache.items())
        peak_payload=max(peak_payload,payload)
        assert payload==counters['cache_bytes']
    elapsed=time.perf_counter()-start
    bad_ids=np.any(returned!=expected,axis=1)
    bad_values=np.any(joined!=expected_values,axis=1)
    np.savez_compressed(OUT/(label+'.npz'),ids=returned,values=joined,latency_seconds=latencies,stage_seconds=stages,cache_hit=hit_mask,cross_k_hit=cross_mask)
    db.execute('pragma wal_checkpoint(TRUNCATE)')
    db.close()
    row={'seed':seed,'clients':clients,'policy':policy,'requests':4096,'cache_payload_cap_bytes':options.cache_bytes,**counters,'wrong_ordered_results':int(bad_ids.sum()),'wrong_joined_values':int(bad_values.sum()),'wall_seconds':elapsed,'p50_ms':float(np.median(latencies)*1000),'p95_ms':float(np.quantile(latencies,.95)*1000),'check_total_seconds':float(stages[:,0].sum()),'native_total_seconds_including_lock_wait':float(stages[:,1].sum()),'join_insert_total_seconds':float(stages[:,2].sum()),'max_epoch_boundary_cache_payload_bytes':peak_payload,'sqlite_bytes':database.stat().st_size,'updates':updates}
    print('RUN',row,flush=True)
    (OUT/(label+'.json')).write_text(json.dumps(row,indent=2))
    if policy!='invalid_unversioned_prefix': assert not bad_ids.any() and not bad_values.any(),label
    return row

def main():
    with h5py.File(ROOT/'data/sift-128-euclidean.hdf5','r') as f:
        queries=np.asarray(f['test'][:],np.float32)
        extra=np.asarray(f['train'][100000:100256],np.float32)
    selected=sorted(range(len(queries)),key=lambda i:hashlib.sha256(str(i).encode()).hexdigest())[256:512]
    rows=[]
    for seed in PROTOCOL['seeds']:
        trace=traces(seed,selected)
        expected,values=reference(trace,queries,extra,seed)
        configurations=[(clients,policy) for clients in [1,4] for policy in ['none','coarse_result_exact_k','ids_exact_k','ids_guarded_budget']]
        if options.invalid_only: configurations=[(1,'invalid_unversioned_prefix')]
        np.random.default_rng(seed).shuffle(configurations)
        for clients,policy in configurations:
            rows.append(run(trace,queries,extra,expected,values,seed,clients,policy))
            (OUT/'partial.json').write_text(json.dumps(rows,indent=2))
    (OUT/'summary.json').write_text(json.dumps({'protocol':PROTOCOL,'rows':rows,'scope':'Closed-loop concurrent readers with epoch-barrier writes, bounded LRU and native library search. Request schedules and revision metadata are synthetic. The search-configuration lock protects the library mutable ef; this is not a production throughput experiment.'},indent=2))

if __name__=='__main__': main()
