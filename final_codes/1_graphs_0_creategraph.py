import dgl
import torch as th

# 边 0->1, 0->2, 0->3, 1->3
u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))
print(g) 
# 图中节点的数量是DGL通过给定的图的边列表中最大的点ID推断所得出的
# Graph(num_nodes=4, num_edges=4,
#       ndata_schemes={}
#       edata_schemes={})

# 获取节点的ID
print(g.nodes())
# tensor([0, 1, 2, 3])
# 获取边的对应端点
print(g.edges())
# (tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]))
# 获取边的对应端点和边ID 
# 
# 显示所有的边和节点
print(g.edges(form='all'))
# (tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]), tensor([0, 1, 2, 3]))

# 如果具有最大ID的节点没有边，在创建图的时候，用户需要明确地指明节点的数量。
g = dgl.graph((u, v), num_nodes=8)
print(g)

# 转换成无向图
bg = dgl.to_bidirected(g)
print(bg.edges())