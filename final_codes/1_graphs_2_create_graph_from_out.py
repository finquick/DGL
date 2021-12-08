import dgl
import torch as th
import scipy.sparse as sp
from matplotlib import pyplot as plt
spmat = sp.rand(100, 100, density=0.05) # 5%非零项
g = dgl.from_scipy(spmat)                   # 来自SciPy
print(g)

# Graph(num_nodes=100, num_edges=500,
#       ndata_schemes={}
#       edata_schemes={})

import networkx as nx
nx_g = nx.path_graph(5) # 一条链路0-1-2-3-4
g = dgl.from_networkx(nx_g) # 来自NetworkX
print(g)

nx.draw(nx_g, with_labels=True, font_weight='bold')  #此处还可参考pyg的demo 做可视化。
plt.savefig("./test.jpg")

plt.show()

# Graph(num_nodes=5, num_edges=8,
#       ndata_schemes={}
#       edata_schemes={})

