"""Symmetry-split exact distance certification.
Level 1 (as qverify): x meets orbit O_c, x = 0 on orbits before c, representative r_c in supp x  (valid by Aut-transitivity on O_c).
Level 2 (new, inside each orbit case): look at S = supp x  within O_c.
  - single:  |S| = 1  ->  x = 0 on O_c minus r_c.
  - pair class t: the A-orbits of unordered pairs in O_c are ordered; t = least class occurring in S.
    Then (after an automorphism) r_c in S and some b in S with {r_c,b} in class t, b ranging over Stab(r_c)-orbit reps of
    B_t = {b : {r_c,b} in class t};  and every pair of class < t is excluded:  x_p + x_q <= 1.
Every logical of weight <= U-1 falls in at least one subcase, so infeasibility of all subcases certifies d >= U.

Usage: python qsplit.py <code> <X|Z> <U> list            -> prints the subcase ids
       python qsplit.py <code> <X|Z> <U> <id,id,...|all> <secs-per-subcase> <threads>
Writes results/split/<slug>__<sector>_U<U>_<ids>.json"""
import sys, os, json, re, time, glob, itertools
import numpy as np, highspy
import qverify, clp_ci, qli_ci

def group_perms(entry, Hx, Hz):
    """Verified automorphism permutations (qubit q -> sigma[q])."""
    n = Hx.shape[1]
    if entry['kind'] in ('clp', 'atb', 'lp'): return list(clp_ci.LAST_PERMS)
    if entry['kind'] == 'qli': return list(qli_ci.LAST_PERMS)
    if entry['kind'] == 'bb':          # translations of Z_l x Z_m on both blocks
        l, m = entry['l'], entry['m']; N = l * m
        out = []
        for a, b in ((1, 0), (0, 1)):
            p = np.array([((i + a) % l) * m + (j + b) % m for i in range(l) for j in range(m)])
            out.append(np.concatenate([p, p + N]))
        return out
    if entry['kind'] == 'tt':          # translations of Z^2 / <(0,alpha),(beta,gamma)>
        alpha, beta, gamma = entry['torus']; N = alpha * beta
        def idx(i, j):
            q, i = divmod(i, beta); j = (j - q * gamma) % alpha; return i * alpha + j
        out = []
        for a, b in ((1, 0), (0, 1)):
            p = np.zeros(N, dtype=int)
            for i in range(beta):
                for j in range(alpha): p[i * alpha + j] = idx(i + a, j + b)
            out.append(np.concatenate([p, p + N]))
        return out
    return []

def closure(gens, n, cap=20000):
    I = np.arange(n); els = {I.tobytes(): I}; frontier = [I]
    while frontier:
        nf = []
        for g in frontier:
            for s in gens:
                h = s[g]
                k = h.tobytes()
                if k not in els:
                    els[k] = h; nf.append(h)
                    if len(els) > cap: raise RuntimeError('group too large')
        frontier = nf
    return list(els.values())

def subcases(blocks, G, n):
    """List of (case_id, dict(fix1=[...], fix0=[...], forbid_pairs=[(p,q),...]))."""
    out = []
    for c, O in enumerate(blocks):
        r = O[0]; zeros = [j for b in blocks[:c] for j in b]; Os = set(O)
        if len(O) == 1 or not G:
            out.append((f'{c}', dict(fix1=[r], fix0=zeros, pairs=[]))); continue
        # pair classes in O
        cls = {}; order = []
        for p, q in itertools.combinations(sorted(O), 2):
            if (p, q) in cls: continue
            cid = len(order); order.append((p, q))
            for g in G:
                a, b = int(g[p]), int(g[q])
                if a in Os and b in Os: cls[(min(a, b), max(a, b))] = cid
        stab = [g for g in G if g[r] == r]
        out.append((f'{c}.s', dict(fix1=[r], fix0=zeros + [j for j in O if j != r], pairs=[])))
        for t in range(len(order)):
            Bt = sorted(b for b in O if b != r and cls[(min(r, b), max(r, b))] == t)
            reps, seen = [], set()
            for b in Bt:
                if b in seen: continue
                reps.append(b); seen |= {int(g[b]) for g in stab}
            forb = [pq for pq, s in cls.items() if s < t]
            for b in reps:
                out.append((f'{c}.{t}.{b}', dict(fix1=[r, b], fix0=zeros, pairs=forb)))
    return out

def solve(H, L, U, sc, secs, threads):
    n = H.shape[1]
    h = highspy.Highs(); h.setOptionValue('output_flag', False)
    h.setOptionValue('time_limit', float(secs)); h.setOptionValue('threads', threads)
    for j in range(n): h.addVar(0, 1)
    nv = [n]; integ = list(range(n))
    def addint(u):
        h.addVar(0, u); nv[0] += 1; return nv[0] - 1
    for row in H:
        s = np.nonzero(row)[0]; t = addint(len(s) // 2); integ.append(t)
        h.addRow(0, 0, len(s) + 1, list(map(int, s)) + [t], [1.0] * len(s) + [-2.0])
    av = []
    for row in L:
        s = np.nonzero(row)[0]; t = addint(len(s) // 2); a = addint(1); integ += [t, a]; av.append(a)
        h.addRow(0, 0, len(s) + 2, list(map(int, s)) + [t, a], [1.0] * len(s) + [-2.0, -1.0])
    h.addRow(1, highspy.kHighsInf, len(av), av, [1.0] * len(av))
    h.addRow(0, U - 1, n, list(range(n)), [1.0] * n)
    for j in range(n): h.changeColCost(j, 1.0)
    for j in integ: h.changeColIntegrality(j, highspy.HighsVarType.kInteger)
    for j in sc['fix1']: h.changeColBounds(int(j), 1.0, 1.0)
    for j in sc['fix0']: h.changeColBounds(int(j), 0.0, 0.0)
    for p, q in sc['pairs']: h.addRow(0, 1, 2, [int(p), int(q)], [1.0, 1.0])
    t0 = time.time(); h.run(); st = h.getModelStatus(); dt = round(time.time() - t0, 1)
    if st == highspy.HighsModelStatus.kInfeasible: return dict(status='infeasible', secs=dt)
    sol = np.array(h.getSolution().col_value[:n]).round().astype(int)
    rec = dict(status='optimal' if st == highspy.HighsModelStatus.kOptimal else str(st).split('.')[-1], secs=dt,
               dual_bound=h.getInfo().mip_dual_bound)
    if sol.sum() > 0 and not (H @ sol % 2).any() and (L @ sol % 2).any() and sol.sum() <= U - 1:
        rec['witness'] = np.nonzero(sol)[0].tolist(); rec['weight'] = int(sol.sum())
    return rec

def resolve(i, D, blocks):
    """Subcase id 'c.t.b' or 'c.t.b@k:bits' (fix the next k free coordinates of orbit c to the bits; exhaustive 2^k split)."""
    base, _, cube = i.partition('@')
    sc = dict(D[base]); sc['fix1'] = list(sc['fix1']); sc['fix0'] = list(sc['fix0'])
    if cube:
        k, bits = map(int, cube.split(':'))
        orb = blocks[int(base.split('.')[0])]
        fixed = set(map(int, sc['fix1'])) | set(map(int, sc['fix0']))
        for r, j in enumerate([j for j in orb if j not in fixed][:k]):
            (sc['fix1'] if (bits >> r) & 1 else sc['fix0']).append(j)
    return sc

def setup(name, sector):
    entry = next(e for f in sorted(glob.glob('codes*.json')) for e in json.load(open(f)) if e['name'] == name)
    Hx, Hz, blocks, entry = qverify.build(entry)
    n = Hx.shape[1]
    perms = group_perms(entry, Hx, Hz)
    from qverify import GF2
    rx, rz = np.linalg.matrix_rank(GF2(Hx % 2)), np.linalg.matrix_rank(GF2(Hz % 2))
    def is_aut(sg):
        inv = np.argsort(sg)
        return (np.linalg.matrix_rank(GF2(np.vstack([Hx, Hx[:, inv]]) % 2)) == rx and
                np.linalg.matrix_rank(GF2(np.vstack([Hz, Hz[:, inv]]) % 2)) == rz)
    assert all(is_aut(np.asarray(p)) for p in perms), 'non-automorphism among symmetry generators'
    G = closure(perms, n) if perms else []
    if blocks is None:
        if G:   # orbits of G
            par = list(range(n))
            for g in perms:
                for q in range(n):
                    a, b = q, int(g[q])
                    while par[a] != a: a = par[a]
                    while par[b] != b: b = par[b]
                    if a != b: par[a] = b
            orb = {}
            for q in range(n):
                a = q
                while par[a] != a: a = par[a]
                orb.setdefault(a, []).append(q)
            blocks = sorted((sorted(v) for v in orb.values()), key=lambda v: v[0])
        else: blocks = [[q] for q in range(n)]
    H, L = (Hz, qverify.logicals(Hz, Hx)) if sector == 'X' else (Hx, qverify.logicals(Hx, Hz))
    return entry, H.astype(int) % 2, L.astype(int) % 2, [list(b) for b in blocks], G

if __name__ == '__main__':
    name, sector, U, what = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    entry, H, L, blocks, G = setup(name, sector)
    SC = subcases(blocks, G, H.shape[1])
    if what == 'list':
        print(json.dumps(dict(n=H.shape[1], group=len(G), orbits=[len(b) for b in blocks], subcases=[i for i, _ in SC]))); sys.exit()
    secs, threads = float(sys.argv[5]), int(sys.argv[6])
    engine = os.environ.get('QS_ENGINE', 'highs')
    ids = [i for i, _ in SC] if what == 'all' else what.split(',')
    D = dict(SC); out = []
    for i in ids:
        sc = resolve(i, D, blocks)
        if engine == 'gurobi':
            import grb
            r = grb.solve(H, L, U - 1, secs, threads, sc['fix1'], sc['fix0'], sc['pairs'])
            rec = dict(id=i, secs=r['secs'], dual_bound=r['dual_bound'],
                       status='UNSAT' if r['status'] == 'infeasible' else ('SAT' if r['witness'] is not None else 'UNKNOWN'))
            if r['witness'] is not None: rec['witness'] = np.nonzero(r['witness'])[0].tolist(); rec['weight'] = int(r['witness'].sum())
        else:
            r = solve(H, L, U, sc, secs, threads)
            rec = dict(id=i, **r); rec['status'] = 'UNSAT' if r['status'] == 'infeasible' else ('SAT' if 'witness' in r else 'UNKNOWN')
        rec['engine'] = engine; out.append(rec); print(json.dumps(rec), flush=True)
    slug = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    os.makedirs('results/sat', exist_ok=True)
    tag = 'all' if what == 'all' else re.sub(r'[^0-9a-z.]+', '_', what)[:60]
    json.dump(dict(name=name, sector=sector, U=U, engine=engine, group=len(G), n_subcases=len(SC), subcase_ids=[i for i, _ in SC], results=out,
                   run_url=f"{os.environ.get('GITHUB_SERVER_URL','')}/{os.environ.get('GITHUB_REPOSITORY','')}/actions/runs/{os.environ.get('GITHUB_RUN_ID','')}"),
              open(f'results/sat/{slug}__{sector}_U{U}_{engine}_{tag}.json', 'w'), indent=1)
