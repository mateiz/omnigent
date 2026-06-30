import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def solve(objective_function, config, best_xs=None):
    bounds = np.array(config['bounds'], dtype=float)
    dim = int(config.get('dim', bounds.shape[0]))
    budget = int(config.get('budget', 100))
    low, high = bounds[:, 0], bounds[:, 1]

    rng = np.random.default_rng()
    all_attempts = []

    def clip(x):
        return np.clip(x, low, high)

    def eval_x(x):
        x = np.asarray(x, dtype=float)
        x = clip(x)
        score = float(objective_function(x))
        all_attempts.append({"x": x.copy(), "score": score})
        return score

    # Incorporate previous attempts if provided (these don't count against current budget)
    prev_points = []
    if best_xs:
        for item in best_xs:
            x = np.asarray(item["x"], dtype=float)
            s = float(item["score"])
            prev_points.append((x, s))
    # We'll also keep a working list of known points for local modeling
    known_X = [x for x, _ in prev_points]
    known_y = [s for _, s in prev_points]

    # Helper: select diverse seeds from known good points + random LHS
    def latin_hypercube(n_samples, dim, low, high):
        # Standard LHS in [0,1]^d then scale
        cut = np.linspace(0, 1, n_samples + 1)
        u = rng.uniform(size=(n_samples, dim))
        a = cut[:n_samples]
        b = cut[1:n_samples + 1]
        rdpoints = u * (b - a)[:, None] + a[:, None]
        for j in range(dim):
            rng.shuffle(rdpoints[:, j])
        return low + rdpoints * (high - low)

    # Determine number of starts and local steps based on budget
    # Reserve some for random/global exploration, some for local refinement
    min_evals_for_local = 5  # to try at least a small local search
    if budget <= 0:
        # No budget to evaluate; return best previous if any, else center
        if prev_points:
            best_prev = min(prev_points, key=lambda t: t[1])
            return {"x": best_prev[0], "score": best_prev[1], "all_attempts": []}
        x0 = (low + high) / 2.0
        return {"x": x0, "score": float("inf"), "all_attempts": []}

    # Start with best known point if available; otherwise sample a point
    if prev_points:
        best_prev_x, best_prev_s = min(prev_points, key=lambda t: t[1])
        best_x = best_prev_x.copy()
        best_s = best_prev_s
    else:
        x0 = rng.uniform(low, high)
        s0 = eval_x(x0)
        best_x, best_s = x0.copy(), s0

    # Compute remaining budget
    remaining = budget - len(all_attempts)
    if remaining <= 0:
        return {"x": best_x, "score": best_s, "all_attempts": all_attempts}

    # Seed pool: top-K from prev + LHS
    seeds = []
    if prev_points:
        # Take top-k diverse previous points
        k = max(1, min(8, int(np.sqrt(remaining)) ))
        prev_sorted = sorted(prev_points, key=lambda t: t[1])[:min(len(prev_points), 50)]
        # Greedy diversity by distance
        selected = []
        for x, s in prev_sorted:
            if not selected:
                selected.append((x, s))
            else:
                dists = cdist([x], [p[0] for p in selected])[0]
                if np.all(dists > 1e-6 * np.linalg.norm(high - low) + 1e-9):
                    selected.append((x, s))
            if len(selected) >= k:
                break
        seeds.extend([x for x, _ in selected])
    # Add LHS samples
    n_lhs = max(0, min(remaining // 3, 10))
    if n_lhs > 0:
        lhs = latin_hypercube(n_lhs, dim, low, high)
        seeds.extend([lhs[i] for i in range(n_lhs)])

    # Ensure we have at least one seed
    if not seeds:
        seeds = [rng.uniform(low, high)]

    # Evaluate seeds (avoid duplicates of prev_points if same x)
    # We'll keep a set of hashes to avoid exact duplicates
    def x_hash(x):
        return tuple(np.round(x, 12))
    seen = set(x_hash(a["x"]) for a in all_attempts)
    for x in seeds:
        if remaining <= 0:
            break
        h = x_hash(x)
        if h in seen:
            continue
        s = eval_x(x)
        seen.add(h)
        remaining -= 1
        if s < best_s:
            best_x, best_s = x.copy(), s

    if remaining <= 0:
        return {"x": best_x, "score": best_s, "all_attempts": all_attempts}

    # Local trust-region searches around a few best from observed (prev + new)
    obs_X = [a["x"] for a in all_attempts] + [x for x, _ in prev_points]
    obs_y = [a["score"] for a in all_attempts] + [s for _, s in prev_points]
    order = np.argsort(obs_y)
    sorted_X = [obs_X[i] for i in order][:min(5, len(order))]

    # Trust-region parameters
    # Scale relative to bounds
    span = np.maximum(1e-12, high - low)
    base_radius = 0.25 * span  # initial radius
    min_radius = 1e-3 * span

    def local_search(x0, steps_budget):
        nonlocal best_x, best_s, remaining
        x = x0.copy()
        radius = base_radius.copy()
        # Simple pattern search within trust region
        directions = np.eye(dim)
        while steps_budget > 0 and remaining > 0:
            improved = False
            for sign in [+1.0, -1.0]:
                for d in directions:
                    if steps_budget <= 0 or remaining <= 0:
                        break
                    candidate = clip(x + sign * radius * d)
                    h = x_hash(candidate)
                    if h in seen:
                        continue
                    s = eval_x(candidate)
                    seen.add(h)
                    remaining -= 1
                    steps_budget -= 1
                    if s < best_s:
                        best_x, best_s = candidate.copy(), s
                    if s < eval_x.cache.get(x_hash(x), np.inf):
                        x = candidate
                        eval_x.cache[x_hash(x)] = s
                        improved = True
                if steps_budget <= 0 or remaining <= 0:
                    break
            if not improved:
                # shrink radius
                radius = np.maximum(min_radius, 0.5 * radius)
                # If minimal radius reached, stop
                if np.allclose(radius, min_radius):
                    break
        return

    # cache to store known score for a point during local search decisions
    eval_x.cache = {}
    for a in all_attempts:
        eval_x.cache[x_hash(a["x"])] = a["score"]
    for x, s in prev_points:
        eval_x.cache[x_hash(x)] = s

    # Allocate remaining between local searches and global random exploration
    n_locals = min(len(sorted_X), max(1, remaining // 10))
    steps_per_local = max(2, remaining // max(1, n_locals + 1))

    for i in range(n_locals):
        if remaining <= 0:
            break
        local_search(sorted_X[i], steps_per_local)

    if remaining <= 0:
        return {"x": best_x, "score": best_s, "all_attempts": all_attempts}

    # Final global exploration: random + perturb around best
    # Mix of Latin hypercube and Gaussian perturbations
    n_global = remaining
    n_gauss = n_global // 2
    n_rand = n_global - n_gauss

    # Gaussian around top few points
    topX = sorted_X if sorted_X else [best_x]
    cov = np.diag((0.1 * span) ** 2)
    for i in range(n_gauss):
        base = topX[i % len(topX)]
        cand = rng.multivariate_normal(mean=base, cov=cov)
        cand = clip(cand)
        h = x_hash(cand)
        if h in seen:
            continue
        s = eval_x(cand)
        seen.add(h)
        remaining -= 1
        if s < best_s:
            best_x, best_s = cand.copy(), s
        if remaining <= 0:
            break

    # Uniform random for the rest
    for _ in range(max(0, remaining)):
        cand = rng.uniform(low, high)
        h = x_hash(cand)
        if h in seen:
            continue
        s = eval_x(cand)
        seen.add(h)
        remaining -= 1
        if s < best_s:
            best_x, best_s = cand.copy(), s
        if remaining <= 0:
            break

    return {"x": best_x, "score": best_s, "all_attempts": all_attempts}