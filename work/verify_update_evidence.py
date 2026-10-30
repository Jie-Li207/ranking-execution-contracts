from pathlib import Path
import argparse,json
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('root',type=Path); args=p.parse_args()
root=args.root
counts={'mechanism_comparisons':0,'false_admissions':0,'far_native_changes':0,'service_runs':0,'service_requests':0,'wrong_ids':0,'wrong_distances':0,'wrong_values':0,'budget_violations':0}
for file in sorted((root/'results/update_confirmation').glob('*/summary.json')):
    for row in json.loads(file.read_text()):
        old=np.load(file.parent/f'before_ef{row["ef"]}.npz'); new=np.load(file.parent/f'after_ef{row["ef"]}_append{row["append"]}.npz')
        assert np.array_equal(old['query_ids'],new['query_ids'])
        changed=np.any(old['ids']!=new['ids'],axis=1)
        far=new['min_inserted_distance']>old['cutoff_float64']+(1e-3 if row['dataset']=='sift' else 1e-5)
        assert np.array_equal(far,new['far'])
        assert int(changed.sum())==row['changed']
        assert int((far&changed).sum())==row['far_changes']
        assert int(new['admitted'].sum())==row['admitted']
        false=int((new['admitted']&changed).sum())
        assert false==row['false_admitted']==0
        exact=root/'results/update_exact'/f'{row["dataset"]}_{row["scale"]}'
        ti=np.load(exact/'exact_before.npz')['ids']; ta=np.load(exact/f'exact_after_{row["append"]}.npz')['ids']
        assert not np.any(np.any(ti!=ta,axis=1)&far)
        counts['mechanism_comparisons']+=len(changed); counts['false_admissions']+=false; counts['far_native_changes']+=int((far&changed).sum())
for group in ['service_confirmation','service_confirmation_matched','service_timing_followup_matched']:
    for file in sorted((root/'results'/group).glob('*/summary.json')):
        rows=json.loads(file.read_text()); refs={}
        for row in rows:
            z=np.load(file.parent/(row['label']+'.npz')); update=row['config']['update']
            if update not in refs:
                source=file.parent/f'oracle_update{update}.npz'
                if not source.exists(): source=root/'results/service_confirmation_matched'/file.parent.name/f'oracle_update{update}.npz'
                oracle=np.load(source); refs[update]=(oracle['ids'],oracle['distances'])
            oi,od=refs[update]; ei=oi[z['epochs'],z['trace']]; ed=od[z['epochs'],z['trace']]
            ids=z['ids']; epoch=z['epochs']; values=np.zeros_like(ids,dtype=np.int32)
            if update:
                mask=ids>=row['scale']; values[mask]=(ids[mask]-row['scale'])//update+1
            for e in range(1,int(epoch.max())+1):
                low=(e*7919)%max(1,row['scale']-256)
                values[(epoch[:,None]>=e)&(ids>=low)&(ids<low+256)]=e
            counts['wrong_ids']+=int(np.any(ids!=ei,axis=1).sum())
            counts['wrong_distances']+=int(np.any(z['distances']!=ed,axis=1).sum())
            counts['wrong_values']+=int(np.any(z['values']!=values,axis=1).sum())
            counts['budget_violations']+=int(row['peak_auxiliary_payload_bytes']>row['config']['bytes'])
            counts['service_runs']+=1; counts['service_requests']+=len(ids)
assert all(counts[k]==0 for k in ['false_admissions','wrong_ids','wrong_distances','wrong_values','budget_violations'])
print(json.dumps({'scope':'Stored numerical evidence verification; fresh source/index execution is a separate documented command.','counts':counts},indent=2))
