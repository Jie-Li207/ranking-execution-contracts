import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
import argparse,json,numpy as np
parser=argparse.ArgumentParser(); parser.add_argument('--wrapper',choices=['native','matched'],default='native'); parser.add_argument('--phase',choices=['confirmation','timing_followup'],default='confirmation'); args=parser.parse_args()
suffix='_matched' if args.wrapper=='matched' else ''
root=Path(os.environ.get('ICDE_UPDATE_ROOT',Path(__file__).resolve().parents[1]))
rng=np.random.default_rng(20261030)
rows=[]
cases=[('sift',100000),('t2i',100000),('t2i',1000000)] if args.phase=='confirmation' else [('t2i',100000),('t2i',1000000)]
for dataset,scale in cases:
    folder=root/('results/service_'+args.phase+suffix)/f'{dataset}_{scale}'
    source=json.loads((folder/'summary.json').read_text())
    for name in sorted({r['config']['name'] for r in source}):
        group=[r for r in source if r['config']['name']==name]
        by={(r['seed'],r['policy']):r for r in group}; seeds=sorted({r['seed'] for r in group})
        row={'dataset':dataset,'scale':scale,'configuration':name,'seeds':len(seeds)}
        for policy in ['none','global','footprint']:
            r=[by[(s,policy)] for s in seeds]
            row[policy]={'wall_median_s':float(np.median([v['wall_seconds'] for v in r])),'p95_median_ms':float(np.median([v['p95_ms'] for v in r])),'hits_mean':float(np.mean([v['hits'] for v in r])),'peak_bytes':max(v['peak_auxiliary_payload_bytes'] for v in r),'writer_max_lag_requests':max(u['completed_at']-u['requested_after'] for v in r for u in v['updates'])}
        for baseline in ['none','global']:
            ratios=np.array([by[(s,baseline)]['wall_seconds']/by[(s,'footprint')]['wall_seconds'] for s in seeds]); boot=np.median(ratios[rng.integers(0,len(seeds),(20000,len(seeds)) )],axis=1)
            row['speedup_vs_'+baseline]={'ratios':ratios.tolist(),'median':float(np.median(ratios)),'ci95':np.quantile(boot,[.025,.975]).tolist()}
        rows.append(row)
(root/('results/service_summary'+('' if args.phase=='confirmation' else '_'+args.phase)+suffix+'.json')).write_text(json.dumps(rows,indent=2))
for row in rows:
    a=row['speedup_vs_none']; b=row['speedup_vs_global']
    print(row['dataset'],row['scale'],row['configuration'],'vs_none',round(a['median'],3),[round(x,3) for x in a['ci95']],'vs_global',round(b['median'],3),[round(x,3) for x in b['ci95']],flush=True)
mechanism=[]
for file in sorted((root/'results/update_confirmation').glob('*/summary.json')): mechanism.extend(json.loads(file.read_text()))
(root/'results/mechanism_summary.json').write_text(json.dumps(mechanism,indent=2))
