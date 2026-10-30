from pathlib import Path
import hashlib,json,requests,os

root=Path(os.environ.get('ICDE_UPDATE_ROOT',Path(__file__).resolve().parents[1]))
folder=root/'work/native_observer/hnswlib'
folder.mkdir(parents=True,exist_ok=True)
expected_hashes={'hnswlib.h': '54403db81f55fd28246114c5c9f683e663a2a55ad64903b56578c73e6db5ad2c', 'hnswalg.h': '0fb2c1b1d3aae1ea959b536f96844f3f3ae6a2f32ce063f2c31f93d84a7e5c32', 'space_l2.h': 'c599d024657896412250b4fde1a95e39f8bc55cccd263f36c8948fa35779edf1', 'space_ip.h': '6a89b5275527fe3592a9b02ea4fc50293fdc68cbd9f7c1ec13683581e1ebbaf7', 'bruteforce.h': 'da1f179e99d8e5a4c39b5db0c681bccf3c63e797aaa4982620519f8e001cb2c6', 'visited_list_pool.h': '8e5bab8d12b5ef26e3b603b656b8a1acba6738498d1d98f44c9fdb1cb0e25110', 'stop_condition.h': '930da73eead77dff0c8c9f3281c5d12250051c90ef98fcfbab8f5f307bdae8d9'}
sources=[]
for name in ['hnswlib.h','hnswalg.h','space_l2.h','space_ip.h','bruteforce.h','visited_list_pool.h','stop_condition.h']:
    url='https://raw.githubusercontent.com/nmslib/hnswlib/v0.8.0/hnswlib/'+name
    original=root/'sources'/name
    if not original.exists():
        r=requests.get(url,timeout=30); r.raise_for_status(); original.write_bytes(r.content)
    sources.append({'file':name,'url':url,'sha256':hashlib.sha256(original.read_bytes()).hexdigest()})
    if sources[-1]['sha256']!=expected_hashes[name]: raise RuntimeError('Source hash mismatch: '+name)
    data=original.read_text()
    if name=='hnswalg.h':
        assert sources[-1]['sha256']=='0fb2c1b1d3aae1ea959b536f96844f3f3ae6a2f32ce063f2c31f93d84a7e5c32'
        data=data.replace('    size_t max_elements_{0};','    std::vector<uint64_t> observed_versions_;\n    static std::vector<tableint>*& observed_reads() {\n        static thread_local std::vector<tableint>* p = nullptr;\n        return p;\n    }\n\n    size_t max_elements_{0};',1)
        for signature in ['linklistsizeint *get_linklist0(tableint internal_id) const {','linklistsizeint *get_linklist(tableint internal_id, int level) const {']:
            assert data.count(signature)==1
            data=data.replace(signature,signature+'\n        if (observed_reads()) observed_reads()->push_back(internal_id);',1)
        before='            if (!is_cur_c_present) {\n                if (sz_link_list_other < Mcurmax) {'
        after='            if (!is_cur_c_present) {\n                std::vector<tableint> observed_before;\n                if (!observed_versions_.empty()) observed_before.assign(data, data + sz_link_list_other);\n                if (sz_link_list_other < Mcurmax) {'
        assert data.count(before)==1; data=data.replace(before,after,1)
        before='                }\n            }\n        }\n\n        return next_closest_entry_point;'
        after='                }\n                if (!observed_versions_.empty() && (getListCount(ll_other) != observed_before.size() || !std::equal(observed_before.begin(), observed_before.end(), data))) ++observed_versions_[selectedNeighbors[idx]];\n            }\n        }\n\n        return next_closest_entry_point;'
        assert data.count(before)==1; data=data.replace(before,after,1)
    (folder/name).write_text(data)
(root/'sources/observer_sources.json').write_text(json.dumps(sources,indent=2))

license_url='https://raw.githubusercontent.com/nmslib/hnswlib/v0.8.0/LICENSE'
license_path=root/'sources/hnswlib_LICENSE.txt'
if not license_path.exists():
    response=requests.get(license_url,timeout=30); response.raise_for_status(); license_path.write_bytes(response.content)
if hashlib.sha256(license_path.read_bytes()).hexdigest()!='b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1': raise RuntimeError('Upstream license hash mismatch')
print('Pinned source observation hooks ready')
