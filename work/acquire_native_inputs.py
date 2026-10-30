from pathlib import Path
import hashlib,json,time,requests

root=Path(__file__).resolve().parents[1]
target=root/'data/sift-128-euclidean.hdf5'
target.parent.mkdir(parents=True,exist_ok=True)
(root/'sources').mkdir(parents=True,exist_ok=True)
url='https://ann-benchmarks.com/sift-128-euclidean.hdf5'
start=time.time()
if not target.exists():
    part=target.with_suffix('.part')
    with requests.get(url,stream=True,timeout=(30,120)) as r:
        r.raise_for_status()
        size=int(r.headers.get('content-length',0))
        print('download',r.status_code,size,flush=True)
        with part.open('wb') as out:
            for i,chunk in enumerate(r.iter_content(1048576)):
                out.write(chunk)
                if i%64==0: print('MiB',i,flush=True)
    part.replace(target)
h=hashlib.sha256()
with target.open('rb') as f:
    for block in iter(lambda:f.read(4194304),b''): h.update(block)
receipt={'url':url,'bytes':target.stat().st_size,'sha256':h.hexdigest(),'seconds':time.time()-start,'authenticated':False,'source':'Official ANN-Benchmarks dataset distribution'}
assert receipt['sha256']=='dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984'
(root/'sources/sift_download.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
