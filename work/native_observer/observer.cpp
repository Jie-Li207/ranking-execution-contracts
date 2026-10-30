#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnswlib/hnswlib.h"
#include <shared_mutex>
#include <algorithm>
#include <memory>

namespace py = pybind11;
using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

class ObservedIndex {
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    mutable std::shared_mutex lock;
    size_t dimension;
public:
    ObservedIndex(std::string path, size_t dim, std::string metric, size_t capacity, size_t ef, bool track_versions=true): dimension(dim) {
        if (metric == "ip") space.reset(new hnswlib::InnerProductSpace(dim));
        else if (metric == "l2") space.reset(new hnswlib::L2Space(dim));
        else throw std::invalid_argument("Unsupported metric");
        index.reset(new hnswlib::HierarchicalNSW<float>(space.get(),path,false,capacity));
        if (index->num_deleted_ != 0) throw std::invalid_argument("Deleted nodes excluded");
        index->setEf(ef);
        if (track_versions) index->observed_versions_.resize(capacity,0);
    }
    py::tuple query(Array q,size_t k,bool observe) {
        if (q.size() != dimension || k == 0) throw std::invalid_argument("Query dimension or budget");
        if (observe && index->observed_versions_.empty()) throw std::invalid_argument("Tracking disabled");
        std::vector<hnswlib::tableint> reads;
        std::vector<uint64_t> versions;
        std::vector<uint64_t> ids(k);
        std::vector<float> distances(k);
        uint64_t entry,level;
        {
            py::gil_scoped_release release;
            std::shared_lock<std::shared_mutex> guard(lock);
            entry=index->enterpoint_node_; level=index->maxlevel_;
            index->observed_reads()=observe ? &reads : nullptr;
            try {
                auto result=index->searchKnn(q.data(),k);
                index->observed_reads()=nullptr;
                if (result.size()!=k) throw std::runtime_error("Insufficient results");
                for (size_t j=k;j>0;--j) {
                    distances[j-1]=result.top().first;
                    ids[j-1]=result.top().second;
                    result.pop();
                }
            } catch (...) { index->observed_reads()=nullptr; throw; }
            if (observe) {
                std::sort(reads.begin(),reads.end());
                reads.erase(std::unique(reads.begin(),reads.end()),reads.end());
                versions.reserve(reads.size());
                for (auto id:reads) versions.push_back(index->observed_versions_[id]);
            }
        }
        return py::make_tuple(py::array_t<uint64_t>(ids.size(),ids.data()),py::array_t<float>(distances.size(),distances.data()),py::array_t<uint32_t>(reads.size(),reads.data()),py::array_t<uint64_t>(versions.size(),versions.data()),entry,level);
    }
    bool valid(py::array_t<uint32_t,py::array::c_style|py::array::forcecast> nodes,py::array_t<uint64_t,py::array::c_style|py::array::forcecast> versions,uint64_t entry,uint64_t level) const {
        if (nodes.size()!=versions.size()) throw std::invalid_argument("Certificate length");
        py::gil_scoped_release release;
        std::shared_lock<std::shared_mutex> guard(lock);
        if (entry!=index->enterpoint_node_ || level!=index->maxlevel_) return false;
        for (size_t j=0;j<size_t(nodes.size());++j) {
            auto id=nodes.data()[j];
            if (id>=index->cur_element_count || versions.data()[j]!=index->observed_versions_[id]) return false;
        }
        return true;
    }
    void append(Array vectors,py::array_t<uint64_t,py::array::c_style|py::array::forcecast> labels) {
        if (vectors.ndim()!=2 || vectors.shape(1)!=dimension || vectors.shape(0)!=labels.size()) throw std::invalid_argument("Append shape");
        py::gil_scoped_release release;
        std::unique_lock<std::shared_mutex> guard(lock);
        if (index->cur_element_count+labels.size()>index->max_elements_) throw std::invalid_argument("Capacity");
        std::unordered_set<uint64_t> seen;
        for (size_t j=0;j<size_t(labels.size());++j) if (index->label_lookup_.count(labels.data()[j]) || !seen.insert(labels.data()[j]).second) throw std::invalid_argument("Existing or duplicate labels excluded");
        for (size_t j=0;j<size_t(labels.size());++j) index->addPoint(vectors.data()+j*dimension,labels.data()[j]);
    }
    void save(std::string path) { std::shared_lock<std::shared_mutex> guard(lock); index->saveIndex(path); }
    uint64_t version_bytes() const { return index->observed_versions_.size()*sizeof(uint64_t); }
};

PYBIND11_MODULE(hnsw_observer,m) {
    py::class_<ObservedIndex>(m,"Index")
    .def(py::init<std::string,size_t,std::string,size_t,size_t,bool>(),py::arg("path"),py::arg("dim"),py::arg("metric"),py::arg("capacity"),py::arg("ef"),py::arg("track_versions")=true)
    .def("query",&ObservedIndex::query)
    .def("valid",&ObservedIndex::valid)
    .def("append",&ObservedIndex::append)
    .def("save",&ObservedIndex::save)
    .def("version_bytes",&ObservedIndex::version_bytes);
}
