import heapq

def lazy_topk(roots, root_scores, get_neighbors, budget):
    pairs = sorted([(int(v), float(s)) for v, s in zip(roots, root_scores)], key=lambda z: (-z[1], z[0]))
    assert budget >= 1 and all((0 <= s <= 1 + 1e-06 for _, s in pairs))
    if budget == 1:
        return ([pairs[0][0]], 0)
    best = {v: (s, -1, v) for v, s in pairs}
    reads = 0
    for parent, score in pairs:
        ranked = heapq.nsmallest(budget, best.values(), key=lambda z: (-z[0], z[1], z[2]))
        threshold = ranked[-1][0] if len(ranked) >= budget else float('-inf')
        if score < threshold:
            break
        neighbors, similarities = get_neighbors(parent)
        reads += 1
        for child, sim in zip(neighbors, similarities):
            child = int(child)
            if child == parent:
                continue
            assert 0 <= sim <= 1 + 1e-06
            w = score * min(float(sim), 1.0)
            candidate = (w, parent, child)
            old = best.get(child)
            if old is None or (-w, parent, child) < (-old[0], old[1], old[2]):
                best[child] = candidate
    ranked = heapq.nsmallest(budget, best.values(), key=lambda z: (-z[0], z[1], z[2]))
    return ([x[2] for x in ranked], reads)
