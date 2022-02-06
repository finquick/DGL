import dgl
import numpy as np
import torch as th
from dgl.nn import ChebConv

g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
feat = th.ones(6, 10)
conv = ChebConv(10, 2, 2)
res = conv(g, feat)
print(res)