"""Gurobi engine for the minimum-weight-logical MILP (same model as qverify.exact_dist / qsplit.solve).
License: Gurobi Web License Service, read from env GRB_WLSACCESSID, GRB_WLSSECRET, GRB_LICENSEID (repository secrets).
    min sum x   s.t.  H x = 2 t   (each check even),   <L_j, x> = 2 s_j + a_j,   sum a_j >= 1,   sum x <= ub,
    x_fix1 = 1, x_fix0 = 0, x_p + x_q <= 1 for excluded pairs;   x, a binary, t, s integer."""
import os, time
import numpy as np

_ENV = None
def env():
    global _ENV
    if _ENV is None:
        import gurobipy as gp
        p = {'OutputFlag': 0}
        if os.environ.get('GRB_WLSACCESSID'):
            p.update(WLSACCESSID=os.environ['GRB_WLSACCESSID'], WLSSECRET=os.environ['GRB_WLSSECRET'],
                     LICENSEID=int(os.environ['GRB_LICENSEID']))
        _ENV = gp.Env(params=p)
    return _ENV

def available():
    try:
        import gurobipy  # noqa
        return bool(os.environ.get('GRB_WLSACCESSID')) or os.environ.get('QV_ENGINE') == 'gurobi'
    except ImportError:
        return False

def solve(H, L, ub, secs, threads, fix1=(), fix0=(), pairs=(), start=None):
    """Returns dict(status in {'infeasible','optimal','timelimit'}, incumbent, dual_bound, witness or None, secs)."""
    import gurobipy as gp
    from gurobipy import GRB
    H = np.asarray(H) % 2; L = np.asarray(L) % 2; n = H.shape[1]
    m = gp.Model(env=env())
    m.Params.TimeLimit = float(secs); m.Params.Threads = int(threads); m.Params.MIPFocus = 3   # push the bound
    x = m.addVars(n, vtype=GRB.BINARY)
    for r in H:
        s = np.nonzero(r)[0]
        t = m.addVar(lb=0, ub=len(s) // 2, vtype=GRB.INTEGER)
        m.addConstr(gp.quicksum(x[int(j)] for j in s) == 2 * t)
    av = []
    for r in L:
        s = np.nonzero(r)[0]
        t = m.addVar(lb=0, ub=len(s) // 2, vtype=GRB.INTEGER); a = m.addVar(vtype=GRB.BINARY); av.append(a)
        m.addConstr(gp.quicksum(x[int(j)] for j in s) == 2 * t + a)
    m.addConstr(gp.quicksum(av) >= 1)
    if ub is not None: m.addConstr(gp.quicksum(x.values()) <= ub)
    for j in fix1: x[int(j)].LB = 1
    for j in fix0: x[int(j)].UB = 0
    nb = {}
    for p, q in pairs: nb.setdefault(int(p), set()).add(int(q)); nb.setdefault(int(q), set()).add(int(p))
    for p, Q in nb.items():   # aggregated exclusion: x_p = 1 forces x_q = 0 on all excluded partners (one row per p)
        m.addConstr(gp.quicksum(x[q] for q in Q) + len(Q) * x[p] <= len(Q))
    if start is not None:
        for j in range(n): x[j].Start = int(start[j])
    m.setObjective(gp.quicksum(x.values()), GRB.MINIMIZE)
    t0 = time.time(); m.optimize(); dt = round(time.time() - t0, 1)
    if m.Status == GRB.INFEASIBLE: return dict(status='infeasible', incumbent=None, dual_bound=None, witness=None, secs=dt)
    wit = None
    if m.SolCount > 0:
        sol = np.array([round(x[j].X) for j in range(n)], dtype=int)
        if sol.sum() > 0 and not (H @ sol % 2).any() and (L @ sol % 2).any(): wit = sol
    st = 'optimal' if m.Status == GRB.OPTIMAL else 'timelimit'
    return dict(status=st, incumbent=int(wit.sum()) if wit is not None else None, dual_bound=float(m.ObjBound), witness=wit, secs=dt)
