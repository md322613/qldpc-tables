"""Exact distance lower bounds by SAT with native XOR (CryptoMiniSat) + automorphism-orbit case splitting.
Question per case: is there x with  H x = 0,  <x, L_j> = 1 for some conjugate logical L_j,  |x| <= w,
x[rep(orbit c)] = 1 and x = 0 on orbits < c ?   UNSAT for every case  =>  d >= w + 1.
Cases can be split further by cubes (fix the next few coordinates of orbit c) and spread over CI jobs.

Usage: python qsat.py <code-name> <sector X|Z> <w> <case-spec> <timeout-s> [threads]
  case-spec: 'all' | 'c' (one orbit case) | 'c:i/m' (i-th of m cubes inside orbit case c)
Writes results/sat/<slug>__<sector>_w<w>_<case>.json  with status UNSAT / SAT(witness) / UNKNOWN."""
import sys, os, json, re, time, itertools, glob
import numpy as np
from pycryptosat import Solver
from pysat.card import CardEnc, EncType
import qverify

def cnf_case(H, L, w, rep, zeros, cube):
    n = H.shape[1]
    var = lambda j: j + 1                       # x_j -> 1..n
    nxt = [n + 1]
    def new():
        nxt[0] += 1; return nxt[0] - 1
    xors, cls = [], []
    for r in H:
        s = np.nonzero(r)[0]
        xors.append(([var(int(j)) for j in s], False))
    ys = []
    for r in L:
        s = np.nonzero(r)[0]; y = new(); ys.append(y)
        xors.append(([var(int(j)) for j in s] + [y], False))     # y = <x, L_j>
    cls.append(ys)                                                # some y_j = 1
    if rep is not None: cls.append([var(rep)])
    for j in zeros: cls.append([-var(int(j))])
    for j, b in cube: cls.append([var(int(j)) if b else -var(int(j))])
    card = CardEnc.atmost(lits=[var(j) for j in range(n)], bound=w, top_id=nxt[0], encoding=EncType.kmtotalizer)
    cls += card.clauses
    return xors, cls

def solve_case(H, L, w, rep, zeros, cube, timeout, threads):
    xors, cls = cnf_case(H, L, w, rep, zeros, cube)
    s = Solver(threads=threads, time_limit=timeout)
    for c in cls: s.add_clause(c)
    for v, rhs in xors: s.add_xor_clause(v, rhs)
    t0 = time.time(); sat, sol = s.solve(); dt = time.time() - t0
    if sat is None: return 'UNKNOWN', None, dt
    if not sat: return 'UNSAT', None, dt
    n = H.shape[1]; x = np.array([1 if sol[j + 1] else 0 for j in range(n)], dtype=np.uint8)
    return 'SAT', x, dt

def cubes(orbit_rest, depth):
    """2^depth assignments of the first `depth` coordinates of the orbit after its representative."""
    pos = list(orbit_rest[:depth])
    return [list(zip(pos, bits)) for bits in itertools.product([0, 1], repeat=len(pos))]

def setup(name, sector):
    entry = next(e for f in sorted(glob.glob('codes*.json')) for e in json.load(open(f)) if e['name'] == name)
    Hx, Hz, blocks, entry = qverify.build(entry)
    n = Hx.shape[1]
    if blocks is None: blocks = [list(range(0, n // 2)), list(range(n // 2, n))]
    if sector == 'X': H, L = Hz, qverify.logicals(Hz, Hx)      # same convention as qverify.run
    else: H, L = Hx, qverify.logicals(Hx, Hz)
    return entry, H % 2, L % 2, [list(b) for b in blocks]

if __name__ == '__main__':
    # python qsat.py <code> <X|Z> <U> <ids|all|list> [timeout threads]   -- U: prove no logical of weight <= U-1
    import qsplit
    name, sector, U, what = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    entry, H, L, blocks, G = qsplit.setup(name, sector)
    SC = qsplit.subcases(blocks, G, H.shape[1]); D = dict(SC)
    if what == 'list':
        print(json.dumps([i for i, _ in SC])); sys.exit()
    timeout, threads = float(sys.argv[5]), int(sys.argv[6])
    ids = [i for i, _ in SC] if what == 'all' else what.split(',')
    out = []
    for i in ids:
        base, _, cube = i.partition('@')
        sc = dict(D[base]); sc['fix1'] = list(sc['fix1']); sc['fix0'] = list(sc['fix0'])
        if cube:   # '@k:i' : fix the next k free coordinates of the case orbit to the bits of i (exhaustive 2^k split)
            k, bits = map(int, cube.split(':'))
            orb = blocks[int(base.split('.')[0])]
            fixed = set(map(int, sc['fix1'])) | set(map(int, sc['fix0']))
            free = [j for j in orb if j not in fixed][:k]
            for r, j in enumerate(free):
                (sc['fix1'] if (bits >> r) & 1 else sc['fix0']).append(j)
        xors, cls = cnf_case(H, L, U - 1, None, sc['fix0'], [(j, 1) for j in sc['fix1']])
        cls += [[-(int(p) + 1), -(int(q) + 1)] for p, q in sc['pairs']]
        s = Solver(threads=threads, time_limit=timeout)
        for c in cls: s.add_clause(c)
        for v, rhs in xors: s.add_xor_clause(v, rhs)
        t0 = time.time(); sat, sol = s.solve(); dt = round(time.time() - t0, 1)
        rec = dict(id=i, status='UNKNOWN' if sat is None else ('SAT' if sat else 'UNSAT'), secs=dt)
        if sat:
            x = np.array([1 if sol[j + 1] else 0 for j in range(H.shape[1])], dtype=np.uint8)
            assert not (H.astype(int) @ x % 2).any() and (L.astype(int) @ x % 2).any() and x.sum() <= U - 1
            rec['witness'] = np.nonzero(x)[0].tolist(); rec['weight'] = int(x.sum())
        out.append(rec); print(json.dumps(rec), flush=True)
    slug = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    os.makedirs('results/sat', exist_ok=True)
    tag = 'all' if what == 'all' else re.sub(r'[^0-9a-z.]+', '_', what)[:60]
    json.dump(dict(name=name, sector=sector, U=U, engine='cryptominisat5 (XOR) + pysat kmtotalizer', group=len(G), n_subcases=len(SC),
                   subcase_ids=[i for i, _ in SC], results=out,
                   run_url=f"{os.environ.get('GITHUB_SERVER_URL','')}/{os.environ.get('GITHUB_REPOSITORY','')}/actions/runs/{os.environ.get('GITHUB_RUN_ID','')}"),
              open(f'results/sat/{slug}__{sector}_U{U}_{tag}.json', 'w'), indent=1)
