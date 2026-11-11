"""Coset-orbit balanced-product codes (Qian-Li, arXiv:2608.08996, Supplementary Information).
Pure Python: host groups are built concretely, qubits/checks are indexed by double cosets K\\G/K.
Balanced product F2[K\\G] (x)_G F2[G/K] = F2[K\\G/K]:
  L(KgK): KxK -> sum_{Kh in K\\KgK} KhxK,   R(KgK): KxK -> sum_{hK in KgK/K} KxhK   (mod 2).
Protographs A (ma x na), B (mb x nb):  Hx = [A (x) I_mb | I_ma (x) B],  Hz = [I_na (x) B^T | A^T (x) I_nb]
with A-entries realized by L and B-entries by R.  For K={e} this is the Lin-Pryadko/Panteleev-Kalachev lifted product."""
import re, itertools, numpy as np, galois
GF2 = galois.GF(2)

# ---------------- concrete groups: elements are hashable tuples ----------------
class Grp:
    def __init__(self, mul, e, gens):
        self.mul, self.e, self.gens = mul, e, dict(gens)
        els = [e]; seen = {e}; i = 0
        while i < len(els):
            for g in self.gens.values():
                h = mul(els[i], g)
                if h not in seen: seen.add(h); els.append(h)
            i += 1
        self.els = els; self.idx = {g: j for j, g in enumerate(els)}
    def pw(self, g, k):
        r = self.e
        for _ in range(k): r = self.mul(r, g)
        return r
    def w(self, *word):
        """word: alternating generator name / exponent, or raw elements."""
        r = self.e; it = iter(word)
        for t in it:
            g = self.gens[t] if isinstance(t, str) else t
            k = next(it) if isinstance(t, str) else 1
            r = self.mul(r, self.pw(g, k))
        return r
    def inv(self, g):
        for h in self.els:
            if self.mul(g, h) == self.e: return h
    def sub(self, gens):
        els = [self.e]; seen = {self.e}; i = 0
        while i < len(els):
            for g in gens:
                h = self.mul(els[i], g)
                if h not in seen: seen.add(h); els.append(h)
            i += 1
        return els

def abelian(*ms, names):
    mul = lambda a, b: tuple((x + y) % m for x, y, m in zip(a, b, ms))
    gens = {nm: tuple(1 if j == i else 0 for j in range(len(ms))) for i, nm in enumerate(names)}
    return Grp(mul, tuple(0 for _ in ms), gens)

def metacyclic(m, t, r, xn='x', sn='s'):
    """Z_m x| Z_t with s x s^-1 = x^r; element (i,j) = x^i s^j."""
    mul = lambda a, b: ((a[0] + b[0] * pow(r, a[1], m)) % m, (a[1] + b[1]) % t)
    return Grp(mul, (0, 0), {xn: (1, 0), sn: (0, 1)})

def dicyclic(nn):
    """Dic_n, order 4n: r^(2n)=e, s^2=r^n, s r s^-1=r^-1; element (i,j)=r^i s^j."""
    M = 2 * nn
    def mul(a, b):
        i = (a[0] + (b[0] if a[1] == 0 else -b[0])) % M; j = a[1] + b[1]
        if j == 2: i = (i + nn) % M; j = 0
        return (i, j)
    return mul, (0, 0), (1, 0), (0, 1)

def direct(*parts):
    """parts: (mul, e) pairs; returns mul, e for the direct product."""
    mul = lambda a, b: tuple(p[0](x, y) for p, x, y in zip(parts, a, b))
    return mul, tuple(p[1] for p in parts)

def matmul(q):
    def mm(A, B):
        return ((A[0]*B[0] + A[1]*B[2]) % q, (A[0]*B[1] + A[1]*B[3]) % q,
                (A[2]*B[0] + A[3]*B[2]) % q, (A[2]*B[1] + A[3]*B[3]) % q)
    return mm

def psl_canon(q):
    def c(A):
        for x in A:
            if x: return A if x <= (q - 1) // 2 else tuple((-y) % q for y in A)
    return c

def perm_mul(a, b):
    """product a*b acting on the right (x^(ab) = (x^a)^b), GAP convention."""
    return tuple(b[a[i]] for i in range(len(a)))

def cyc(*cs, deg=6):
    p = list(range(deg))
    for c in cs:
        for i in range(len(c)): p[c[i] - 1] = c[(i + 1) % len(c)] - 1
    return tuple(p)

# ---------------- the 27 host groups / codes ----------------
def M(*e): return tuple(e)          # 2x2 matrix row-major

def build_group(spec):
    t = spec['group']
    if t == 'Z84:29Z4': G = metacyclic(84, 4, 29)
    elif t == 'Z41:Z8': G = metacyclic(41, 8, 38)
    elif t == 'Z31:Z6': G = metacyclic(31, 6, 26)
    elif t == 'M27xZ7':
        m1 = lambda a, b: ((a[0] + b[0] * pow(4, a[1], 9)) % 9, (a[1] + b[1]) % 3)
        mul = lambda a, b: (m1(a[:2], b[:2]) + ((a[2] + b[2]) % 7,))
        G = Grp(mul, (0, 0, 0), {'x': (1, 0, 0), 's': (0, 1, 0), 'u': (0, 0, 1)})
    elif t == 'Dic5xDic5':
        mul1, e1, r, s = dicyclic(5); mul, e = direct((mul1, e1), (mul1, e1))
        G = Grp(mul, e, {'r1': (r, e1), 's1': (s, e1), 'r2': (e1, r), 's2': (e1, s)})
    elif t == 'Dic4':
        mul1, e1, r, s = dicyclic(4); G = Grp(mul1, e1, {'r': r, 's': s})
    elif t == 'HeZ3xZ7':
        def mul(a, b): return ((a[0]+b[0]) % 3, (a[1]+b[1]) % 3, (a[2]+b[2]+a[0]*b[1]) % 3, (a[3]+b[3]) % 7)
        G = Grp(mul, (0, 0, 0, 0), {'x': (1, 0, 0, 0), 'y': (0, 1, 0, 0), 'u': (0, 0, 0, 1)})
        x, y = G.gens['x'], G.gens['y']; z = G.w(x, y, G.inv(x), G.inv(y)); G.gens['z'] = z
    elif t == 'Z8wrZ2' or t == 'Z17^2:swapZ2':
        m = 8 if t == 'Z8wrZ2' else 17
        def mul(a, b):
            b0, b1 = (b[1], b[0]) if a[2] else (b[0], b[1])
            return ((a[0] + b0) % m, (a[1] + b1) % m, (a[2] + b[2]) % 2)
        gn = ('x1', 'x2') if m == 8 else ('x', 'y')
        G = Grp(mul, (0, 0, 0), {gn[0]: (1, 0, 0), gn[1]: (0, 1, 0), 'sigma': (0, 0, 1)})
    elif t == 'D6:Z8':
        def dm(a, b):   # D6 element (i,j)=r^i s^j
            return ((a[0] + (b[0] if a[1] == 0 else -b[0])) % 6, (a[1] + b[1]) % 2)
        def th(d, z):   # theta^z: r->r, s->r^3 s
            return ((d[0] + 3 * d[1] * (z % 2)) % 6, d[1])
        mul = lambda a, b: (dm(a[:2], th(b[:2], a[2])) + ((a[2] + b[2]) % 8,))
        G = Grp(mul, (0, 0, 0), {'r': (1, 0, 0), 's': (0, 1, 0), 'u': (0, 0, 1)})
    elif t in ('SL23:Z12', 'SL23:Z16', 'GL23xZ4'):
        mm = matmul(3); m, P = {'SL23:Z12': (12, M(1, 0, 0, 2)), 'SL23:Z16': (16, M(1, 1, 2, 1)), 'GL23xZ4': (4, M(1, 0, 0, 1))}[t]
        mul = lambda a, b: (mm(a[0], b[0]), (a[1] + b[1]) % m)
        gens = {'y': (P, 1), 'S1': (M(1, 1, 0, 1), 0), 'S2': (M(1, 0, 1, 1), 0)}
        if t == 'GL23xZ4': gens['S3'] = (M(2, 0, 0, 1), 0)
        G = Grp(mul, (M(1, 0, 0, 1), 0), gens); G.P = P
    elif t == 'A6xZ2':
        mul = lambda a, b: (perm_mul(a[0], b[0]), (a[1] + b[1]) % 2)
        G = Grp(mul, (tuple(range(6)), 0), {'a': (cyc((1, 2, 3)), 0), 'b': (cyc((2, 3, 4, 5, 6)), 0), 'u': (tuple(range(6)), 1)})
    elif t in ('PSL2_11', 'PSL2_13'):
        q = 11 if t == 'PSL2_11' else 13; mm = matmul(q); c = psl_canon(q)
        G = Grp(lambda a, b: c(mm(a, b)), M(1, 0, 0, 1), {'S': M(0, q - 1, 1, 0), 'T': M(1, 1, 0, 1)}); G.c = c
    else:
        ab = {'Z12xZ48': ((12, 48), 'xy'), 'Z4xZ42': ((4, 42), 'xy'), 'Z2xZ2xZ42': ((2, 2, 42), 'xyz'), 'Z13xZ9': ((13, 9), 'xy'),
              'Z31xZ6': ((31, 6), 'xy'), 'Z85': ((85,), 'x'), 'Z195': ((195,), 'x'), 'Z39xZ5': ((39, 5), 'xy')}[t]
        G = abelian(*ab[0], names=list(ab[1]))
    return G

def elem(G, spec, term):
    """term: string of space-separated factors: e | gen | gen^k | [a,b,c,d] (2x2 matrix) | (1,2,3)(4,5) (perm of 1..6)."""
    r = G.e
    for f in term.split():
        if f == 'e': continue
        if f.startswith('['):
            mat = tuple(int(v) for v in f.strip('[]').split(','))
            g = G.c(mat) if hasattr(G, 'c') else (mat, 0)
        elif f.startswith('('):
            g = (cyc(*[tuple(int(v) for v in c.split(',')) for c in re.findall(r'\(([^)]*)\)', f)]), 0)
        else:
            nm, _, k = f.partition('^'); g = G.pw(G.gens[nm], int(k) if k else 1)
        r = G.mul(r, g)
    return r

def double_cosets(G, K):
    Ki = [G.idx[k] for k in K]; N = len(G.els)
    mt = np.array([[G.idx[G.mul(a, b)] for b in G.els] for a in G.els], dtype=np.int32)   # mt[a,b]=ab
    dc = -np.ones(N, dtype=np.int64); reps = []
    for g in range(N):
        if dc[g] >= 0: continue
        orb = {int(mt[mt[k1, g], k2]) for k1 in Ki for k2 in Ki}
        for h in orb: dc[h] = len(reps)
        reps.append(g)
    return mt, Ki, dc, reps

def LR(G, mt, Ki, dc, reps, g, side):
    """Hecke operator of the double coset KgK on F2[K\\G/K]."""
    D = len(reps); Mx = np.zeros((D, D), dtype=np.uint8)
    dbl = {int(mt[mt[k1, g], k2]) for k1 in Ki for k2 in Ki}
    if side == 'L':   # right cosets Kh inside KgK
        cos = {};
        for h in dbl: cos.setdefault(frozenset(int(mt[k, h]) for k in Ki), h)
        for c, x in enumerate(reps):
            for h in cos.values(): Mx[dc[mt[h, x]], c] ^= 1
    else:             # left cosets hK inside KgK
        cos = {}
        for h in dbl: cos.setdefault(frozenset(int(mt[h, k]) for k in Ki), h)
        for c, x in enumerate(reps):
            for h in cos.values(): Mx[dc[mt[x, h]], c] ^= 1
    return Mx

def assemble(spec):
    G = build_group(spec)
    K = G.sub([elem(G, spec, t) for t in spec['K']]) if spec['K'] else [G.e]
    mt, Ki, dc, reps = double_cosets(G, K); D = len(reps)
    def ent(entry, side):
        Mx = np.zeros((D, D), dtype=np.uint8)
        for t in entry: Mx ^= LR(G, mt, Ki, dc, reps, G.idx[elem(G, spec, t)], side)
        return Mx
    A = [[ent(e, 'L') for e in row] for row in spec['A']]; B = [[ent(e, 'R') for e in row] for row in spec['B']]
    ma, na, mb, nb = len(A), len(A[0]), len(B), len(B[0])
    # Hx = [A (x) I_mb | I_ma (x) B],  Hz = [I_na (x) B^T | A^T (x) I_nb]   (Panteleev-Kalachev layout)
    n = (na * mb + ma * nb) * D
    Hx = np.zeros((ma * mb * D, n), dtype=np.uint8); Hz = np.zeros((na * nb * D, n), dtype=np.uint8)
    q1 = lambda j, m: (j * mb + m) * D; q2 = lambda i, l: na * mb * D + (i * nb + l) * D
    for i in range(ma):
        for m in range(mb):
            r = (i * mb + m) * D
            for j in range(na): Hx[r:r+D, q1(j, m):q1(j, m)+D] ^= A[i][j]
            for l in range(nb): Hx[r:r+D, q2(i, l):q2(i, l)+D] ^= B[m][l]
    for j in range(na):
        for l in range(nb):
            r = (j * nb + l) * D
            for m in range(mb): Hz[r:r+D, q1(j, m):q1(j, m)+D] ^= B[m][l].T
            for i in range(ma): Hz[r:r+D, q2(i, l):q2(i, l)+D] ^= A[i][j].T
    # candidate symmetries: left/right translation by normalizer elements (well defined on K\G/K), same on every block
    Kset = set(Ki); Nidx = [a for a in range(len(G.els)) if {int(mt[mt[a, k], 0]) for k in Ki} and
                            {int(mt[a, k]) for k in Ki} == {int(mt[k, a]) for k in Ki}]
    perms = []
    for a in Nidx[:4000]:
        for side in 'LR':
            p = np.array([dc[mt[a, x]] if side == 'L' else dc[mt[x, a]] for x in reps])
            if len(set(p.tolist())) == D: perms.append(p)
    nblocks = n // D
    info = dict(order=len(G.els), K=len(K), D=D, normalizer=len(Nidx), shape=[ma, na, mb, nb])
    return Hx, Hz, nblocks, D, perms, info

def orbits(Hx, Hz, D, nblocks, perms, cap=100000):
    n = Hx.shape[1]; rx = np.linalg.matrix_rank(GF2(Hx)); rz = np.linalg.matrix_rank(GF2(Hz))
    keep = []; par = list(range(n))
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for p in perms[:cap]:
        sigma = np.concatenate([b * D + p for b in range(nblocks)])
        if all(f(q) == f(int(sigma[q])) for q in range(n)): continue    # adds nothing
        inv = np.argsort(sigma)
        if np.linalg.matrix_rank(GF2(np.vstack([Hx, Hx[:, inv]]))) == rx and np.linalg.matrix_rank(GF2(np.vstack([Hz, Hz[:, inv]]))) == rz:
            keep.append(sigma)
            for q in range(n):
                a, b = f(q), f(int(sigma[q]))
                if a != b: par[a] = b
    global LAST_PERMS; LAST_PERMS = keep
    orb = {}
    for q in range(n): orb.setdefault(f(q), []).append(q)
    return sorted((sorted(v) for v in orb.values()), key=lambda v: v[0]), len(keep)
