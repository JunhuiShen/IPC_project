#!/usr/bin/env python3
"""obj2off.py in.obj out.off  --  triangle OBJ -> OFF for TetGen."""
import sys
import numpy as np

V, F = [], []
for ln in open(sys.argv[1]):
    if ln.startswith('v '):
        V.append([float(x) for x in ln.split()[1:4]])
    elif ln.startswith('f '):
        i = [int(t.split('/')[0]) for t in ln.split()[1:]]
        i = [k - 1 if k > 0 else len(V) + k for k in i]
        F += [[i[0], i[k], i[k + 1]] for k in range(1, len(i) - 1)]
V, F = np.asarray(V), np.asarray(F, np.int64)
with open(sys.argv[2], 'w') as f:
    f.write("OFF\n%d %d 0\n" % (len(V), len(F)))
    np.savetxt(f, V, fmt="%.10g %.10g %.10g")
    np.savetxt(f, np.c_[np.full(len(F), 3), F], fmt="%d %d %d %d")
print("%s: %d verts, %d tris -> %s" % (sys.argv[1], len(V), len(F), sys.argv[2]))
