import dgl
import torch as th
g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0])) # 6个节点，4条边
print(g)
# Graph(num_nodes=6, num_edges=4,
#       ndata_schemes={}
#       edata_schemes={})
g.ndata['x'] = th.ones(g.num_nodes(), 3)               # 长度为3的节点特征
g.edata['x'] = th.ones(g.num_edges(), dtype=th.int32)  # 标量整型特征
print(g)
# Graph(num_nodes=6, num_edges=4,
#       ndata_schemes={'x' : Scheme(shape=(3,), dtype=torch.float32)}
#       edata_schemes={'x' : Scheme(shape=(,), dtype=torch.int32)})
# 不同名称的特征可以具有不同形状
g.ndata['y'] = th.randn(g.num_nodes(), 5)
print(g.ndata['x'][1])                 # 获取节点1的特征
# tensor([1., 1., 1.])
print(g.edata['x'][th.tensor([0, 3])] ) # 获取边0和3的特征

# tensor([1, 1], dtype=torch.int32)

#为边添加权重，做成边特征值。
edges = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
weights = th.tensor([0.1, 0.6, 0.9, 0.7])  # 每条边的权重
g = dgl.graph(edges)
g.edata['w'] = weights  # 将其命名为 'w'
print(g)