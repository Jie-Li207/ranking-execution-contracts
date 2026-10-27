import hashlib
import json
import numpy as np

def compare_outputs(reference, candidate, query_ids, depth):
    a = np.asarray(reference)
    b = np.asarray(candidate)
    if a.ndim != 2 or a.shape != b.shape or len(query_ids) != len(a) or (not 1 <= depth <= a.shape[1]):
        raise ValueError('Aligned query rows and valid common output depth are required')
    ordered = np.any(a[:, :depth] != b[:, :depth], axis=1)
    sets = np.any(np.sort(a[:, :depth], axis=1) != np.sort(b[:, :depth], axis=1), axis=1)
    return {'queries': len(a), 'depth': depth, 'ordered_changes': int(ordered.sum()), 'set_changes': int(sets.sum()), 'witness_ids': [str(query_ids[i]) for i in np.flatnonzero(ordered)], 'status': 'falsified_on_inputs' if ordered.any() else 'not_falsified_on_inputs'}

def dependency_key(source_revision, parameters, gallery_ids, gallery_vectors, query_ids=None, query_vectors=None):
    if (query_ids is None) != (query_vectors is None):
        raise ValueError('Cohort identifiers and vectors must both be supplied')
    h = hashlib.sha256(json.dumps({'source': source_revision, 'parameters': parameters}, sort_keys=True, separators=(',', ':')).encode())
    for value in [gallery_ids, gallery_vectors, query_ids, query_vectors]:
        if value is None:
            continue
        a = np.ascontiguousarray(value)
        if a.dtype.hasobject:
            raise ValueError('Use fixed-width ID arrays')
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()

def rewrite_decision(report, proof_guard=None):
    if report['ordered_changes']:
        return 'reject_rewrite'
    if proof_guard is True:
        return 'admit_with_proven_guard'
    return 'no_universal_admission_from_finite_tests'
