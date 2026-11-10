"""Coset two-block codes Q_G^H(a,b) (Aydin-Tamo-Barg, arXiv:2606.17268, Construction 1): Hx=[L(a)|R(b)], Hz=[R(b)^T|L(a)^T].
H = Filtered(AllSubgroups(G), H -> not IsNormal(G,H))[s]; a, b index LeftCosets(G, Core(G,H)) and LeftCosets(N_G(H), H) (paper Table 2 convention)."""
import json, subprocess, numpy as np
def gap_build(e):
    go, gi = e['G']; s = e['s']; a = e['a']; b = e['b']
    sc = f'''SetPrintFormattingStatus("*stdout*",false);; G:=SmallGroup({go},{gi});; L:=Filtered(AllSubgroups(G), H -> not IsNormal(G,H));; H:=L[{s}];;
C:=Core(G,H);; LC:=LeftCosets(G,C);; N:=Normalizer(G,H);; LCN:=LeftCosets(N,H);;
A:=List({a},i->Representative(LC[i]));; B:=List({b},i->Representative(LCN[i]));;
reps:=List(LeftCosets(G,H),Representative);; m:=Length(reps);;
idx:=function(y) local i; for i in [1..m] do if reps[i]^-1*y in H then return i; fi; od; return fail; end;;
gens:=Concatenation(GeneratorsOfGroup(G),GeneratorsOfGroup(N),GeneratorsOfGroup(Centre(G)));;
Print("{{\\"m\\":",m,",\\"A\\":",List(A,g->List(reps,x->idx(g*x))),",\\"B\\":",List(B,g->List(reps,x->idx(x*g))),
",\\"gens\\":[",JoinStringsWithSeparator(List(gens,g->Concatenation("{{\\"L\\":",String(List(reps,x->idx(g*x))),",\\"R\\":",String(List(reps,x->idx(x*g))),"}}")),","),"]}}\\n");
QUIT;'''
    r = subprocess.run(['gap', '-q', '-o', '4g'], input=sc, capture_output=True, text=True, timeout=3600)
    return json.loads(r.stdout.replace(' ', '').replace('\n', ''))
def pm(p, m):
    M = np.zeros((m, m), dtype=np.uint8)
    for i, j in enumerate(p): M[j - 1, i] ^= 1
    return M
def assemble(D):
    m = D['m']; A = sum(pm(p, m).astype(int) for p in D['A']) % 2; B = sum(pm(p, m).astype(int) for p in D['B']) % 2
    return np.hstack([A, B]).astype(np.uint8), np.hstack([B.T, A.T]).astype(np.uint8), 2
