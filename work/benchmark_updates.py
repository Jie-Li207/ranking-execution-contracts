import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager,nullcontext
import argparse,hashlib,json,sqlite3,sys,threading,time
import numpy as np
from audit_updates import load,native,build,canonical,root,previous
import hnsw_observer

class RWLock:
    def __init__(self):
        self.c=threading.Condition(); self.readers=0; self.writing=False; self.waiting=0
    @contextmanager
    def read(self):
        with self.c:
            while self.writing or self.waiting: self.c.wait()
            self.readers+=1
        try: yield
        finally:
            with self.c: self.readers-=1; self.c.notify_all()
    @contextmanager
    def write(self,reserved=False):
        with self.c:
            if not reserved: self.waiting+=1
            while self.writing or self.readers: self.c.wait()
            self.waiting-=1; self.writing=True
        try: yield
        finally:
            with self.c: self.writing=False; self.c.notify_all()
    def reserve(self):
        with self.c: self.waiting+=1
    def cancel(self):
        with self.c: self.waiting=0; self.c.notify_all()

def entry_bytes(v):
    return 24+sum(x.nbytes for x in v[:4])+(16 if len(v)>2 else 0)
def trace_for(seed,rho,window,requests,pool):
    rng=np.random.default_rng(seed); order=rng.permutation(pool); cursor=0; trace=[]
    for j in range(requests):
        if j and rng.random()<rho: qi=trace[int(rng.integers(max(0,j-window),j))]
        else: qi=int(order[cursor%pool]); cursor+=1
        trace.append(qi)
    return np.array(trace,np.int32)
def run(args,config,seed,policy,queries,extra,path,metric,dim,out):
    label=f'{config["name"]}_{seed}_{policy}'
    target=out/(label+'.json')
    if target.exists(): return json.loads(target.read_text())
    trace=trace_for(seed,config['rho'],config['window'],args.requests,len(queries))
    count=args.requests; update=config['update']; clients=config['clients']; cap=config['bytes']
    adapter=policy=='footprint' or args.wrapper=='matched'
    idx=hnsw_observer.Index(str(path),dim,metric,args.scale+2000,128,policy=='footprint') if adapter else native(path,metric,dim,args.scale,128)
    version_bytes=idx.version_bytes() if policy=='footprint' else 0
    if version_bytes>=cap: raise RuntimeError('Insufficient budget for index version state')
    dbpath=out/(label+'.sqlite')
    if dbpath.exists(): raise RuntimeError('Incomplete database exists: '+str(dbpath))
    setup=sqlite3.connect(dbpath); setup.execute('pragma journal_mode=WAL'); setup.execute('create table record(id integer primary key,revision integer not null)'); setup.executemany('insert into record values(?,0)',((i,) for i in range(args.scale))); setup.commit(); setup.close()
    gate=RWLock(); native_slots=threading.Semaphore(2); cache_lock=threading.Lock(); schedule=threading.Condition(); cache=OrderedDict()
    join_slots=threading.Semaphore(config.get('join_slots',clients))
    state={'epoch':0,'index_epoch':0,'completed':0,'next':0,'abort':False,'payload':0,'peak':0,'hits':0,'checks':0,'admitted_after_update':0,'evictions':0}
    epochs=np.zeros(count,np.int32); ids=np.full((count,10),-1,np.int64); values=np.full((count,10),-1,np.int32); dists=np.zeros((count,10),np.float32); lat=np.zeros(count); stages=np.zeros((count,4)); hits=np.zeros(count,bool); checked_mask=np.zeros(count,bool); updates=[]
    sql='select id,revision from record where id in ('+','.join('?' for _ in range(10))+')'
    def worker():
        conn=sqlite3.connect(dbpath); conn.execute('pragma query_only=ON')
        try:
            while True:
                with schedule:
                    if state['abort']: break
                    step=state['next']; state['next']+=1
                if step>=count: break
                qi=int(trace[step]); t=time.perf_counter()
                with gate.read():
                    acquired=time.perf_counter(); epoch=state['epoch']; index_epoch=state['index_epoch']; v=None
                    if policy!='none':
                        with cache_lock:
                            old=cache.get(qi)
                            if old is not None:
                                candidate,validated=old
                                ok=policy=='global' or validated==index_epoch
                                if not ok:
                                    state['checks']+=1; checked_mask[step]=True; ok=idx.valid(*candidate[2:])
                                    if ok: state['admitted_after_update']+=1; cache[qi]=(candidate,index_epoch)
                                if ok: v=candidate; hits[step]=True; state['hits']+=1; cache.move_to_end(qi)
                    checked=time.perf_counter()
                    if v is None:
                        with native_slots:
                            if adapter:
                                result=idx.query(queries[qi],10,policy=='footprint')
                                if policy!='footprint': result=result[:2]
                            else:
                                ni,nd=idx.knn_query(queries[qi],k=10,num_threads=1); result=(ni[0],nd[0])
                        v=(*canonical(result[0],result[1]),*result[2:])
                    searched=time.perf_counter()
                    with join_slots:
                        records=dict(conn.execute(sql,tuple(int(i) for i in v[0])).fetchall())
                        joined=np.array([records[int(i)] for i in v[0]],np.int32)
                    if policy!='none' and not hits[step]:
                        with cache_lock:
                            old=cache.get(qi)
                            if old is not None: state['payload']-=entry_bytes(old[0])
                            cache[qi]=(v,index_epoch); cache.move_to_end(qi); state['payload']+=entry_bytes(v)
                            while state['payload']+version_bytes>cap:
                                _,removed=cache.popitem(last=False); state['payload']-=entry_bytes(removed[0]); state['evictions']+=1
                            state['peak']=max(state['peak'],state['payload']+version_bytes)
                    end=time.perf_counter(); epochs[step]=epoch; ids[step]=v[0]; dists[step]=v[1]; values[step]=joined; lat[step]=end-t; stages[step]=[acquired-t,checked-acquired,searched-checked,end-searched]
                with schedule:
                    state['completed']+=1
                    if state['completed']%256==0 and state['completed']<count: gate.reserve()
                    schedule.notify_all()
        except BaseException:
            with schedule: state['abort']=True; schedule.notify_all()
            gate.cancel()
            raise
        finally: conn.close()
    def writer():
        conn=sqlite3.connect(dbpath)
        try:
            for epoch,threshold in enumerate(range(256,count,256),1):
                with schedule:
                    while state['completed']<threshold and not state['abort']: schedule.wait()
                    if state['abort']: break
                start=time.perf_counter()
                with gate.write(reserved=True):
                    acquired=time.perf_counter(); begin=(epoch-1)*update
                    if update:
                        if adapter: idx.append(extra[begin:begin+update],np.arange(args.scale+begin,args.scale+begin+update,dtype=np.uint64))
                        else: idx.add_items(extra[begin:begin+update],np.arange(args.scale+begin,args.scale+begin+update),num_threads=1)
                        conn.executemany('insert into record values(?,?)',((i,epoch) for i in range(args.scale+begin,args.scale+begin+update)))
                    low=(epoch*7919)%max(1,args.scale-256)
                    conn.execute('update record set revision=? where id>=? and id<?',(epoch,low,low+256)); conn.commit()
                    state['epoch']=epoch
                    if update:
                        state['index_epoch']+=1
                        if policy=='global':
                            with cache_lock: cache.clear(); state['payload']=0
                    completed_at=state['completed']
                updates.append({'epoch':epoch,'requested_after':threshold,'completed_at':completed_at,'queue_seconds':acquired-start,'transaction_seconds':time.perf_counter()-acquired})
        except BaseException:
            with schedule: state['abort']=True; schedule.notify_all()
            gate.cancel()
            raise
        finally: conn.close()
    cpu0=time.process_time(); t0=time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients+1) as pool:
        wf=pool.submit(writer); readers=[pool.submit(worker) for _ in range(clients)]
        for f in readers: f.result()
        wf.result()
    elapsed=time.perf_counter()-t0
    cpu_seconds=time.process_time()-cpu0
    assert state['completed']==count and not state['abort']
    np.savez_compressed(out/(label+'.npz'),trace=trace,epochs=epochs,ids=ids,distances=dists,values=values,latency_seconds=lat,stages_seconds=stages,cache_hit=hits,certificate_checked=checked_mask)
    row={'label':label,'dataset':args.dataset,'scale':args.scale,'seed':seed,'policy':policy,'config':config,'requests':count,'wall_seconds':elapsed,'process_cpu_seconds':cpu_seconds,'throughput':count/elapsed,'p50_ms':float(np.median(lat)*1000),'p95_ms':float(np.quantile(lat,.95)*1000),'p99_ms':float(np.quantile(lat,.99)*1000),'stages_seconds':stages.sum(axis=0).tolist(),'hits':state['hits'],'native_calls':count-state['hits'],'certificate_checks':state['checks'],'admitted_after_update':state['admitted_after_update'],'peak_auxiliary_payload_bytes':state['peak'],'version_bytes':version_bytes,'evictions':state['evictions'],'updates':updates,'verification':'pending independent fresh replay'}
    target.write_text(json.dumps(row,indent=2)); print({k:row[k] for k in ['label','wall_seconds','p95_ms','hits','admitted_after_update']},flush=True)
    del idx
    for suffix in ['', '-wal','-shm']:
        file=Path(str(dbpath)+suffix)
        if file.exists(): file.unlink()
    return row

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['sift','t2i'],required=True); p.add_argument('--scale',type=int,default=100000); p.add_argument('--phase',choices=['development','confirmation','timing_followup'],default='development'); p.add_argument('--requests',type=int,default=4096); p.add_argument('--protocol',type=Path); p.add_argument('--wrapper',choices=['native','matched'],default='native'); args=p.parse_args()
    base,allq,metric,dim=load(args.dataset,args.scale)
    order=sorted(range(len(allq)),key=lambda i:hashlib.sha256(str(i).encode()).hexdigest())
    if args.phase=='development': take=np.array(order[:256])
    else:
        boundary=4352 if args.dataset=='sift' else 2304
        seen={hashlib.sha256(np.asarray(allq[i],np.float32).tobytes()).hexdigest() for i in order[:boundary]}; selected=[]
        for i in order[boundary:]:
            h=hashlib.sha256(np.asarray(allq[i],np.float32).tobytes()).hexdigest()
            if h in seen: continue
            seen.add(h); selected.append(i)
            if len(selected)==4096: break
        take=np.array(selected); assert len(take)==4096
    queries=np.asarray(allq[take],np.float32); extra=np.asarray(base[args.scale:args.scale+1024],np.float32)
    path=build(args.dataset,args.scale,20261030,base,metric,dim)
    out=root/'results'/('service_'+args.phase+('_matched' if args.wrapper=='matched' else ''))/f'{args.dataset}_{args.scale}'; out.mkdir(parents=True,exist_ok=True)
    np.save(out/'query_ids.npy',take)
    if args.protocol: protocol=json.loads(args.protocol.read_text()); configs=protocol['configurations']; seeds=protocol['seeds']
    else: configs=[{'name':'dev','rho':.5,'window':1024,'clients':4,'update':16,'bytes':4194304}]; seeds=[2026103001]
    rows=[]
    for config in configs:
        for seed in seeds:
            policies=['none','global','footprint']; np.random.default_rng(seed).shuffle(policies)
            for policy in policies: rows.append(run(args,config,seed,policy,queries,extra,path,metric,dim,out))
    rows=[json.loads(f.read_text()) for f in sorted(out.glob('*.json')) if f.name.endswith(('_none.json','_global.json','_footprint.json'))]
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__': main()
