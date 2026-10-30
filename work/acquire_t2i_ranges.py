from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,struct,time,requests,argparse,os

root=Path(os.environ.get('ICDE_UPDATE_ROOT',Path(__file__).resolve().parents[1]))
base='https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/'
expected_hashes={'t2i_queries.fbin': '4b9aabc321ca42a674d5854d1f95347deb902c489c7b49b34e38960f60dad784', 't2i_base_100k.fbin': '37363822718086a1e2e0ac0ab7ca73b0747537aba27aa3df4635b3ca3b463f40', 't2i_base_1m.fbin': '43870fb200e6204628790904526da19c356d62a9856d1c377d416c7a6431d8f8'}

def acquire(remote,local,count):
    target=root/'data'/local
    receipt=target.with_suffix('.json')
    if target.exists() and receipt.exists():
        h=hashlib.sha256()
        with target.open('rb') as f:
            for b in iter(lambda:f.read(4194304),b''): h.update(b)
        if h.hexdigest()!=expected_hashes[local]: raise RuntimeError('Existing data hash mismatch')
        return
    need=8+count*200*4
    chunks=root/'data'/('chunks_'+local)
    chunks.mkdir(exist_ok=True)
    size=1024*1024
    def get(start):
        end=min(start+size,need)-1
        path=chunks/str(start)
        if path.exists() and path.stat().st_size==end-start+1: return path
        prefix=root/'data/t2i_base_100k.fbin'
        if local=='t2i_base_1m.fbin' and prefix.exists() and end<prefix.stat().st_size:
            with prefix.open('rb') as f:
                f.seek(start); path.write_bytes(f.read(end-start+1))
            return path
        for attempt in range(4):
            try:
                r=requests.get(base+remote,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(15,30))
                r.raise_for_status()
                assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/') and len(r.content)==end-start+1
                path.write_bytes(r.content)
                return path
            except Exception as e:
                print('retry',remote,start,attempt,str(e)[:120],flush=True)
                if attempt==3: raise
    t=time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths=[]
        for j,path in enumerate(pool.map(get,range(0,need,size))):
            paths.append(path)
            if j%16==0: print(local,j*size,round(time.time()-t,1),flush=True)
    h=hashlib.sha256()
    with target.open('wb') as f:
        for path in paths:
            b=path.read_bytes(); f.write(b); h.update(b)
    with target.open('rb') as f: n,d=struct.unpack('<II',f.read(8))
    assert d==200 and n>=count
    if h.hexdigest()!=expected_hashes[local]: raise RuntimeError('Downloaded data hash mismatch')
    receipt.write_text(json.dumps({'url':base+remote,'status':206,'bytes_downloaded':need,'original_header_n':n,'d':d,'prefix_rows':count,'sha256_downloaded_bytes':h.hexdigest(),'header_preserved':True,'official_protocol':'https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py','metric':'inner product','seconds':time.time()-t,'all_chunk_content_ranges_verified':True},indent=2))
    for path in paths: path.unlink()
    chunks.rmdir()
    print('Finished',local,need,flush=True)
p=argparse.ArgumentParser(); p.add_argument('--million',action='store_true'); args=p.parse_args()
acquire('query.public.100K.fbin','t2i_queries.fbin',100000)
acquire('base.1B.fbin','t2i_base_1m.fbin' if args.million else 't2i_base_100k.fbin',1001024 if args.million else 101024)
