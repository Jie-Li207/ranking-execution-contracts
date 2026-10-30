from pathlib import Path
import argparse,hashlib,itertools,json,sys
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('root',nargs='?',default=str(Path(__file__).resolve().parents[1]))
args=p.parse_args()
root=Path(args.root)
checks=[]
def check(name,condition):
    if not condition: raise AssertionError(name)
    checks.append(name)
def load(path): return json.loads(path.read_text())
def ordered(d,ids):
    order=np.lexsort((ids,d))
    return d[order],ids[order]
guard={}
for folder in sorted((root/'results/native_ann').glob('*confirmation')):
    report=load(folder/'summary.json')
    with np.load(folder/'outputs.npz') as z:
        selected=np.array(sorted(range(10000),key=lambda i:hashlib.sha256(str(i).encode()).hexdigest())[256:2304])
        check(folder.name+'/query_ids',np.array_equal(selected,z['query_ids']))
        check(folder.name+'/development_disjoint',not set(z['query_ids'])&set(np.load(folder.with_name(folder.name.replace('confirmation','development'))/'outputs.npz')['query_ids']))
        for row in report['records']:
            ef=row['ef']
            if 'k' not in row:
                check(folder.name+f'/batch{ef}',np.array_equal(z[f'ef{ef}_k10_ids'],z[f'ef{ef}_reverse_ids']) and np.array_equal(z[f'ef{ef}_k10_ids'],z[f'ef{ef}_partition_ids']))
                continue
            k=row['k']; a=z[f'ef{ef}_k{k}_ids']; b=z[f'ef{ef}_k100_ids'][:,:k]
            check(folder.name+f'/prefix{ef},{k}',int(np.any(a!=b,axis=1).sum())==row['versus_k100']['ordered_prefix_changes'])
            da=z[f'ef{ef}_k{k}_distances']; db=z[f'ef{ef}_k100_distances'][:,:k]
            check(folder.name+f'/strict{ef},{k}',int(np.any((a!=b)&(abs(da.astype(float)-db.astype(float))>1e-6),axis=1).sum())==row['versus_k100']['strict_prefix_changes'])
            recall=np.mean([len(set(x)&set(y[:k]))/k for x,y in zip(a,z['ground_truth'])])
            check(folder.name+f'/recall{ef},{k}',abs(recall-row['recall'])<1e-12)
        if folder.name.startswith('hnswlib'):
            for ef in [16,64,128]:
                admitted=errors=0
                for k,K in itertools.combinations([1,5,10,20,50,100],2):
                    for di,ii,dk,ik in zip(z[f'ef{ef}_k{K}_distances'],z[f'ef{ef}_k{K}_ids'],z[f'ef{ef}_k{k}_distances'],z[f'ef{ef}_k{k}_ids']):
                        d0,i0=ordered(di,ii); _,i1=ordered(dk,ik)
                        ok=max(ef,k)==max(ef,K) and d0[k-1]<d0[k]
                        admitted+=int(ok)
                        errors+=int(ok and not np.array_equal(i0[:k],i1))
                guard[(folder.name,ef)]=(admitted,errors)
for row in load(root/'results/native_budget_guard.json')['rows']:
    check(str((row['workload'],row['ef']))+'/guard',guard[(row['workload'],row['ef'])]==(row['admitted'],0))
service=[]
for dirname in ['budget_service_v2','budget_service_bytes65536','budget_service_bytes65536_invalid']:
    folder=root/'results'/dirname
    rows=load(folder/'summary.json')['rows']
    check(dirname+'/complete',len(rows)==(3 if dirname.endswith('invalid') else 24))
    for row in rows:
        label=f"{row['seed']}_clients{row['clients']}_{row['policy']}"
        with np.load(folder/(label+'.npz')) as archive: z={k:archive[k] for k in archive.files}
        with np.load(folder/f"reference_{row['seed']}.npz") as archive: ref={k:archive[k] for k in archive.files}
        idsbad=np.any(z['ids']!=ref['ids'],axis=1)
        valuesbad=np.any(z['values']!=ref['values'],axis=1)
        check(dirname+'/'+label+'/counts',int(idsbad.sum())==row['wrong_ordered_results'] and int(valuesbad.sum())==row['wrong_joined_values'])
        expected=np.full_like(ref['values'],-1)
        for step,(_,k,ef) in enumerate(ref['trace']):
            ids=ref['ids'][step,:k]
            updates=sum(step>=v for v in [1024,3072])-(ids>=100000)
            expected[step,:k]=updates*(ids%8==0)
        check(dirname+'/'+label+'/birth_epochs',np.array_equal(expected,ref['values']))
        check(dirname+'/'+label+'/hits',int(z['cache_hit'].sum())==row['hits'] and int(z['cross_k_hit'].sum())==row['prefix_hits'])
        check(dirname+'/'+label+'/native_calls',row['native_calls']==4096-row['hits'])
        check(dirname+'/'+label+'/latency',np.all(np.isfinite(z['latency_seconds'])) and np.all(z['latency_seconds']>=0) and abs(np.quantile(z['latency_seconds'],.95)*1000-row['p95_ms'])<1e-9)
        if not dirname.endswith('invalid'): check(dirname+'/'+label+'/equality',not idsbad.any() and not valuesbad.any())
        if row.get('cache_payload_cap_bytes',0): check(dirname+'/'+label+'/byte_cap',row['peak_cache_bytes']<=65536)
        service.append({'folder':dirname,'label':label,'id_errors':int(idsbad.sum()),'value_errors':int(valuesbad.sum()),'value_errors_with_identical_ids':int((valuesbad&~idsbad).sum())})
corrected=load(root/'results/service_reference_correction.json')
check('reference_correction_record',bool(corrected))
report={'checks_passed':len(checks),'guard_admitted':sum(v[0] for v in guard.values()),'guard_false_admissions':sum(v[1] for v in guard.values()),'service_runs':service,'scope':'Independent numerical verification of stored native outputs, not a new encoder/index execution or an external replication.'}
(root/'native_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='service_runs'},indent=2))
