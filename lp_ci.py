"""Two-block group algebra codes LP[a,b] (Lin-Pryadko, arXiv:2306.16400): Hx=[L(a)|R(b)], Hz=[R(b)^T|L(a)^T] over the group
given by the paper's presentation; IdGroup is checked against the paper's (order, GAP number)."""
import json, subprocess, numpy as np
def gap_word(w):  # 's r^6' style: list of [gen, exp]
    return '*'.join(f'{g}^{e}' for g, e in w) or 'One(G)'
def gap_build(e):
    gens = e['gens']; rels = e['rels']; a = e['a']; b = e['b']
    sc = f'''SetPrintFormattingStatus("*stdout*",false);; F:=FreeGroup({','.join('"%s"' % g for g in gens)});; {' '.join(f'{g}:=F.{i+1};;' for i, g in enumerate(gens))}
Q:=F/[{rels}];; iso:=IsomorphismPermGroup(Q);; G:=Image(iso);; {' '.join(f'{g}:=Image(iso,Q.{i+1});;' for i, g in enumerate(gens))}
els:=AsSortedList(G);; m:=Length(els);; idx:=function(y) return Position(els,y); end;;
A:=[{','.join(gap_word(w) for w in a)}];; B:=[{','.join(gap_word(w) for w in b)}];;
gg:=Concatenation(GeneratorsOfGroup(G),GeneratorsOfGroup(Centre(G)));;
Print("{{\\"m\\":",m,",\\"id\\":",IdGroup(G),",\\"A\\":",List(A,g->List(els,x->idx(g*x))),",\\"B\\":",List(B,g->List(els,x->idx(x*g))),
",\\"gens\\":[",JoinStringsWithSeparator(List(gg,g->Concatenation("{{\\"L\\":",String(List(els,x->idx(g*x))),",\\"R\\":",String(List(els,x->idx(x*g))),"}}")),","),"]}}\\n");
QUIT;'''
    sc = sc.replace('{', '{').replace('}', '}')
    r = subprocess.run(['gap', '-q', '-o', '4g'], input=sc, capture_output=True, text=True, timeout=int(__import__('os').environ.get('GAP_TIMEOUT','3600')))
    return json.loads(r.stdout.replace(' ', '').replace('\n', ''))
def pm(p, m):
    M = np.zeros((m, m), dtype=np.uint8)
    for i, j in enumerate(p): M[j - 1, i] ^= 1
    return M
def assemble(D):
    m = D['m']; A = sum(pm(p, m).astype(int) for p in D['A']) % 2; B = sum(pm(p, m).astype(int) for p in D['B']) % 2
    return np.hstack([A, B]).astype(np.uint8), np.hstack([B.T, A.T]).astype(np.uint8), 2
