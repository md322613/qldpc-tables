"""Exact minimum-distance certification for two-block CSS codes: abelian BB-type and coset lifted product (via GAP).
Usage: python qverify.py <code-name> <time-limit-seconds-per-case> <threads>
Reads codes.json, writes results/<slug>.json with d, certification status, per-case dual bounds and a witness."""
import sys, json, re, time, os
import numpy as np, galois, highspy
GF2 = galois.GF(2)

def parse(s):
    out = []
    for t in s.split('+'):
        if t == '1': out.append((0, 0)); continue
        a = re.search(r'x(-?\d+)', t); b = re.search(r'y(-?\d+)', t)
        out.append((int(a.group(1)) if a else 0, int(b.group(1)) if b else 0))
    return out

def group_code(alpha, beta, gamma, A, B):
    """Z^2 / <(0,alpha),(beta,gamma)>; plain l x m torus is (alpha,beta,gamma)=(m,l,0). Hx=[A|B], Hz=[B^T|A^T]."""
    N = alpha * beta
    def idx(i, j):
        q, i = divmod(i, beta); j = (j - q * gamma) % alpha; return i * alpha + j
    def mat(P):
        M = np.zeros((N, N), dtype=np.uint8)
        for i in range(beta):
            for j in range(alpha):
                for a, b in P: M[idx(i + a, j + b), i * alpha + j] ^= 1
        return M
    Am, Bm = mat(A), mat(B)
    return np.hstack([Am, Bm]) % 2, np.hstack([Bm.T, Am.T]) % 2

def rank2(M): return int(np.linalg.matrix_rank(GF2(M % 2)))

def logicals(H1, H2):
    """Representatives of ker(H2) modulo rowspace(H1)."""
    K = GF2(H2).null_space(); cur = GF2(H1).row_space(); L = []
    for v in K:
        t = np.vstack([cur, v])
        if np.linalg.matrix_rank(t) > cur.shape[0]: cur = t; L.append(np.array(v, dtype=np.uint8))
    return np.array(L, dtype=np.uint8)

def exact_dist(H, Lc, blocks, tl, threads, deadline=None):
    H = np.asarray(H) % 2; L = np.asarray(Lc) % 2; n = H.shape[1]
    best = None; wit = None; cases = []
    for ci, blk in enumerate(blocks):
        h = highspy.Highs(); h.setOptionValue('output_flag', False)
        lim = float(tl)
        if deadline is not None:
            lim = max(60.0, min(lim, (deadline - time.time()) / (len(blocks) - ci)))
        h.setOptionValue('time_limit', lim); h.setOptionValue('threads', threads)
        for j in range(n): h.addVar(0, 1)
        nv = [n]; integ = list(range(n))
        def addint(u):
            h.addVar(0, u); nv[0] += 1; return nv[0] - 1
        for r in H:
            s = np.nonzero(r)[0]; t = addint(len(s) // 2); integ.append(t)
            h.addRow(0, 0, len(s) + 1, list(map(int, s)) + [t], [1.0] * len(s) + [-2.0])
        av = []
        for r in L:
            s = np.nonzero(r)[0]; t = addint(len(s) // 2); a = addint(1); integ += [t, a]; av.append(a)
            h.addRow(0, 0, len(s) + 2, list(map(int, s)) + [t, a], [1.0] * len(s) + [-2.0, -1.0])
        h.addRow(1, highspy.kHighsInf, len(av), av, [1.0] * len(av))
        for j in range(n): h.changeColCost(j, 1.0)
        for j in integ: h.changeColIntegrality(j, highspy.HighsVarType.kInteger)
        h.changeColBounds(int(blk[0]), 1.0, 1.0)
        for b in blocks[:ci]:
            for j in b: h.changeColBounds(int(j), 0.0, 0.0)
        if best is not None: h.addRow(0, best - 1, n, list(range(n)), [1.0] * n)
        t0 = time.time(); h.run(); st = h.getModelStatus()
        if st == highspy.HighsModelStatus.kInfeasible:
            cases.append(dict(case=ci, status='infeasible', secs=round(time.time() - t0, 1))); continue
        sol = np.array(h.getSolution().col_value[:n]).round().astype(int)
        opt = st == highspy.HighsModelStatus.kOptimal
        cases.append(dict(case=ci, status='optimal' if opt else str(st), incumbent=int(sol.sum()),
                          dual_bound=h.getInfo().mip_dual_bound, secs=round(time.time() - t0, 1)))
        if sol.sum() > 0 and not (H @ sol % 2).any() and (best is None or sol.sum() < best):
            best = int(sol.sum()); wit = sol
    lbs = []
    for c in cases:
        if c['status'] == 'infeasible' or c['status'] == 'optimal': lbs.append(best)
        else: lbs.append(int(np.ceil(c['dual_bound'] - 1e-6)))
    lo = min(lbs) if best is not None else None
    return best, lo, wit, cases

def run(entry, tl, threads):
    blocks = None
    if entry['kind'] == 'atb':
        import atb_ci, clp_ci
        D = atb_ci.gap_build(entry); Hx, Hz, nb = atb_ci.assemble(D)
        blocks, ngens = clp_ci.automorphism_orbits(Hx, Hz, D['m'], nb, D['gens'])
        entry = dict(entry, automorphism_generators=ngens, orbit_sizes=[len(o) for o in blocks])
    elif entry['kind'] == 'clp':
        import clp_ci
        m, perm, gens = clp_ci.gap_build(entry)
        Hx, Hz, nb = clp_ci.assemble(entry, m, perm)
        blocks, ngens = clp_ci.automorphism_orbits(Hx, Hz, m, nb, gens)
        entry = dict(entry, automorphism_generators=ngens, orbit_sizes=[len(o) for o in blocks])
    elif entry['kind'] == 'bb':
        Hx, Hz = group_code(entry['m'], entry['l'], 0, parse(entry['A']), parse(entry['B']))
    else:
        al, be, ga = entry['torus']; P = parse(entry['f'])
        Hx, Hz = group_code(al, be, ga, P, [(-a, -b) for a, b in P])
    assert not ((Hx.astype(int) @ Hz.T.astype(int)) % 2).any(), 'checks do not commute'
    n = Hx.shape[1]; k = n - rank2(Hx) - rank2(Hz)
    out = dict(entry, n=n, k=k, w=int(Hx.sum(1).max()), qdeg=int(Hx.sum(0).max() + Hz.sum(0).max()))
    if blocks is None: blocks = [range(0, n // 2), range(n // 2, n)]
    budget = float(os.environ.get('QV_BUDGET', '17400'))   # total seconds for this code (job limit is 350 min)
    t_start = time.time()
    for ti, (typ, H, Lc) in enumerate((('X', Hz, logicals(Hz, Hx)), ('Z', Hx, logicals(Hx, Hz)))):
        deadline = t_start + budget * (ti + 1) / 2
        d, lo, w, cases = exact_dist(H, Lc, blocks, tl, threads, deadline)
        ok = bool(w is not None and not (H @ w % 2).any() and (Lc @ w % 2).any())
        out['d' + typ] = dict(upper=d, lower=lo, exact=(d is not None and lo == d), witness_ok=ok,
                              witness=np.nonzero(w)[0].tolist() if w is not None else None, cases=cases)
    u = min(out['dX']['upper'], out['dZ']['upper']); l = min(out['dX']['lower'], out['dZ']['lower'])
    out['d'] = dict(upper=u, lower=l, exact=(u == l))
    return out

if __name__ == '__main__':
    name, tl, threads = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    import glob
    entry = next(e for f in sorted(glob.glob('codes*.json')) for e in json.load(open(f)) if e['name'] == name)
    os.makedirs('results', exist_ok=True)
    t = time.time(); res = run(entry, tl, threads); res['wall_secs'] = round(time.time() - t, 1)
    res['env'] = dict(highspy=highspy.__version__ if hasattr(highspy, '__version__') else 'n/a', runner=os.environ.get('RUNNER_NAME', 'local'),
                      run_url=f"{os.environ.get('GITHUB_SERVER_URL','')}/{os.environ.get('GITHUB_REPOSITORY','')}/actions/runs/{os.environ.get('GITHUB_RUN_ID','')}")
    slug = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    json.dump(res, open(f'results/{slug}.json', 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('name', 'n', 'k', 'd', 'wall_secs')}))
