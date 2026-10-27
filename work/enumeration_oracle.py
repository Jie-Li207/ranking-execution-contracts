import numpy as np

def enumerate_roots(roots, scores, edges, budgets, gold=None, tolerance=1e-07):
    roots = list(map(int, roots))
    scores = np.asarray(scores, float)
    nr = len(roots)
    edges = list(edges)
    budgets = list(budgets)
    if not 1 <= nr <= 10 or len(set(roots)) != nr or scores.shape != (nr,):
        raise ValueError('Require 1 to 10 distinct roots and one score per root')
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError('Root scores must be finite and nonnegative')
    if any((u not in roots or v in roots or (not np.isfinite(w)) or (w < 0) for u, v, w in edges)):
        raise ValueError('Require edges from roots to disjoint children with finite nonnegative weights')
    if any((not isinstance(B, (int, np.integer)) or B < 1 for B in budgets)):
        raise ValueError('Budgets must be positive integers')
    children = sorted({int(v) for u, v, w in edges})
    ci = {v: i for i, v in enumerate(children)}
    ri = {u: i for i, u in enumerate(roots)}
    nc = len(children)
    W = np.full((nr, nc), -np.inf)
    assert np.all(scores >= 0) and all((w >= 0 for u, v, w in edges))
    for u, v, w in edges:
        W[ri[u], ci[v]] = max(W[ri[u], ci[v]], w)
    size = 1 << nr
    best = np.full((size, nc), -np.inf)
    rootval = np.zeros(size)
    count = np.zeros(size, dtype=int)
    for mask in range(1, size):
        low = mask & -mask
        j = low.bit_length() - 1
        prev = mask ^ low
        best[mask] = np.maximum(best[prev], W[j])
        rootval[mask] = rootval[prev] + scores[j]
        count[mask] = count[prev] + 1
    sorted_weights = np.sort(best, axis=1)[:, ::-1]
    prefix = np.column_stack([np.zeros(size), np.cumsum(np.where(np.isfinite(sorted_weights), sorted_weights, 0), axis=1)])
    masks = np.arange(size)
    result = []
    if gold in ci:
        gw = best[:, ci[gold]]
        other = np.sort(np.delete(best, ci[gold], axis=1), axis=1)[:, ::-1]
        other_prefix = np.column_stack([np.zeros(size), np.cumsum(np.where(np.isfinite(other), other, 0), axis=1)])
    for B in budgets:
        slots = np.clip(B - count, 0, nc)
        val = rootval + prefix[masks, slots]
        val[count > B] = -np.inf
        arg = int(np.argmax(val))
        opt = float(val[arg])
        sel = [u for j, u in enumerate(roots) if arg & 1 << j]
        order = np.lexsort((np.asarray(children), -best[arg]))
        sel += [children[j] for j in order[:int(slots[arg])] if np.isfinite(best[arg, j])]
        forced = float(np.max(val[masks & 1 != 0]))
        can = None
        must = None
        if gold is not None:
            if gold in ri:
                has = masks & 1 << ri[gold] != 0
                yes = float(np.max(val[has]))
                no = float(np.max(val[~has]))
            elif gold in ci:
                take = np.clip(B - count - 1, 0, max(nc - 1, 0))
                yes_values = rootval + gw + other_prefix[masks, take]
                yes_values[(count >= B) | ~np.isfinite(gw)] = -np.inf
                yes = float(np.max(yes_values))
                take = np.clip(B - count, 0, max(nc - 1, 0))
                no_values = rootval + other_prefix[masks, take]
                no_values[count > B] = -np.inf
                no = float(np.max(no_values))
            else:
                yes = -np.inf
                no = opt
            can = opt - yes <= tolerance
            must = opt - no > tolerance
        result.append({'B': B, 'value': opt, 'set': sel, 'forced_first_value': forced, 'strict_first_exclusion': opt - forced > tolerance, 'gold_can_appear': can, 'gold_must_appear': must, 'root_subsets': size})
    return result
