import warnings, itertools, heapq
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

def solve_tree(nodes, edges, B, required=(), forbidden=(), time_limit=5):
    nodes = sorted(set(nodes) - {-1})
    N = len(nodes)
    M = len(edges)
    where = {v: i for i, v in enumerate(nodes)}
    n = N + 3 * M
    rows = []
    lower = []
    upper = []

    def row(items, lo=-np.inf, hi=np.inf):
        rows.append(items)
        lower.append(lo)
        upper.append(hi)
    row({i: 1 for i in range(N)}, hi=B)
    row({**{i: -1 for i in range(N)}, **{N + j: 1 for j in range(M)}}, 0, 0)
    for j, (u, v, w) in enumerate(edges):
        for z in [u, v]:
            if z != -1:
                row({N + j: 1, where[z]: -1}, hi=0)
        row({N + M + 2 * j: 1, N + j: -B}, hi=0)
        row({N + M + 2 * j + 1: 1, N + j: -B}, hi=0)
    for v in nodes:
        items = {where[v]: -1}
        for j, (a, b, _) in enumerate(edges):
            if b == v:
                items[N + M + 2 * j] = 1
                items[N + M + 2 * j + 1] = -1
            if a == v:
                items[N + M + 2 * j] = -1
                items[N + M + 2 * j + 1] = 1
        row(items, 0, 0)
    A = lil_matrix((len(rows), n))
    for i, dd in enumerate(rows):
        for j, val in dd.items():
            A[i, j] = val
    lo = np.zeros(n)
    hi = np.r_[np.ones(N + M), np.full(2 * M, B)]
    c = np.r_[np.zeros(N), -np.array([w for _, _, w in edges]), np.zeros(2 * M)]
    for v in required:
        lo[where[v]] = 1
    for v in forbidden:
        hi[where[v]] = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = milp(c, integrality=np.r_[np.ones(N + M), np.zeros(2 * M)], bounds=Bounds(lo, hi), constraints=LinearConstraint(A.tocsr(), lower, upper), options={'time_limit': time_limit, 'mip_rel_gap': 0, 'mip_abs_gap': 0, 'mip_feasibility_tolerance': 1e-09, 'primal_feasibility_tolerance': 1e-09, 'dual_feasibility_tolerance': 1e-09, 'threads': 1})
    selected = [v for v in nodes if result.x is not None and result.x[where[v]] > 0.5]
    return (result, selected)

def prim_prefix(edges, B):
    adj = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    seen = {-1}
    heap = [(-w, -1, v) for v, w in adj.get(-1, [])]
    heapq.heapify(heap)
    selected = []
    utility = 0
    while heap and len(selected) < B:
        nw, u, v = heapq.heappop(heap)
        if v in seen:
            continue
        seen.add(v)
        selected.append(v)
        utility -= nw
        for z, w in adj.get(v, []):
            if z not in seen:
                heapq.heappush(heap, (-w, v, z))
    return (selected, utility)

def verify_small(seed=41, cases=100):
    import networkx as nx
    rng = np.random.default_rng(seed)
    max_error = 0
    for case in range(cases):
        N = int(rng.integers(3, 8))
        nodes = list(range(N))
        edges = [(-1, j, float(rng.random())) for j in range(N)]
        edges += [(u, v, float(rng.random())) for u in range(N) for v in range(u + 1, N) if rng.random() < 0.3]
        B = int(rng.integers(1, N + 1))
        best = 0
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        for k in range(1, B + 1):
            for sel in itertools.combinations(nodes, k):
                H = G.subgraph([-1, *sel])
                if nx.is_connected(H):
                    best = max(best, nx.maximum_spanning_tree(H).size(weight='weight'))
        res, sel = solve_tree(nodes, edges, B)
        assert res.status == 0
        err = abs(-res.fun - best)
        max_error = max(max_error, err)
        assert err < 1e-07
    return {'cases': cases, 'max_objective_error': max_error, 'reference': 'exhaustive node subsets + NetworkX maximum spanning tree, synthetic only'}
if __name__ == '__main__':
    print(verify_small())
