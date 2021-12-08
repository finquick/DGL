

import dgl
import torch as th

# # 创建一个具有3种节点类型和3种边类型的异构图
# graph_data = {
#    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
#    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
#    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
# }
# g = dgl.heterograph(graph_data)
# print(g.ntypes)
# # ['disease', 'drug', 'gene']
# print(g.etypes)
# # ['interacts', 'interacts', 'treats']
# print(g.canonical_etypes)
# # [('drug', 'interacts', 'drug'),
# #  ('drug', 'interacts', 'gene'),
# #  ('drug', 'treats', 'disease')]
# print(g)

# print(g.metagraph().edges())

# #查看节点总数、每类节点个数、每类节点号
# print(g.num_nodes(), g.num_nodes("drug"), g.num_nodes("gene"), g.num_nodes("disease"), 
#     g.nodes("drug"), g.nodes("gene"), g.nodes("disease"))

# #异质图节点特征、边特征赋值
# #g.nodes[‘node_type’].data[‘feat_name’] 和 g.edges[‘edge_type’].data[‘feat_name’] 。

# # 设置/获取"drug"类型的节点的"hv"特征
# g.nodes['drug'].data['hv'] = th.ones(3, 1)
# print(g.nodes['drug'].data['hv'])
# # tensor([[1.],
# #         [1.],
# #         [1.]])
# # 设置/获取"treats"类型的边的"he"特征
# g.edges['treats'].data['he'] = th.zeros(1, 1)
# print(g.edges['treats'].data['he'])

# #只有一类节点或者边，属性赋值不用单独指定类型
# g = dgl.heterograph({
#    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
#    ('drug', 'is similar', 'drug'): (th.tensor([0, 1]), th.tensor([2, 3]))
# })

# print(g.nodes())
# # tensor([0, 1, 2, 3])
# # 设置/获取单一类型的节点或边特征，不必使用新的语法
# g.ndata['hv'] = th.ones(4, 1)
# print(g.ndata['hv'])



#########################################
##异质图看转化为同质图进行研究#

g = dgl.heterograph({
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))})
g.nodes['drug'].data['hv'] = th.zeros(3, 1)
g.nodes['disease'].data['hv'] = th.ones(3, 1)
g.edges['interacts'].data['he'] = th.zeros(2, 1)
g.edges['treats'].data['he'] = th.zeros(1, 2)

# # 默认情况下不进行特征合并 只是将网络结构拿过来了 这样最保险不会有任何问题
# hg = dgl.to_homogeneous(g)
# print('hv' in hg.ndata)
# print(hg.nodes())
# print(hg)
# # False
# # tensor([0, 1, 2, 3, 4, 5])
# # Graph(num_nodes=6, num_edges=3,
# #       ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64), '_TYPE': Scheme(shape=(), dtype=torch.int64)}
# #       edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64), '_TYPE': Scheme(shape=(), dtype=torch.int64)})

# 异质图转换成同质图过程中如果遇到维度不同无法转变时的处理方法。
# 拷贝边的特征
# hg = dgl.to_homogeneous(g, edata=['he'], ndata=['hv']) #由于he特征长度有1和2两种，所以不行
# # 1、只拷贝节点特征
# hg = dgl.to_homogeneous(g, ndata=['hv'])##没问题#heterogeneous
# print(hg)
# #2、修改he属性维度也可以 以下代码为修改了65行代码统一维度后结果为：g.edges['treats'].data['he'] = th.zeros(1, 1)
# print(hg)
# Graph(num_nodes=6, num_edges=3,
#       ndata_schemes={'hv': Scheme(shape=(1,), dtype=torch.float32), '_ID': Scheme(shape=(), dtype=torch.int64), '_TYPE': Scheme(shape=(), dtype=torch.int64)}
#       edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32), '_ID': Scheme(shape=(), dtype=torch.int64), '_TYPE': Scheme(shape=(), dtype=torch.int64)})
# 3、抛弃特征维度不一样的边或者点，只留特征维度相同的节点和特征维度相同的边构成子图
#
# sub_g = dgl.edge_type_subgraph(g, [('drug', 'interacts', 'drug')])
# h_sub_g = dgl.to_homogeneous(sub_g)
# print(h_sub_g)

# # 分别查看转换完后同质图里节点在异质图里原始的类型和ID号
# # 异构图中节点类型的顺序
# print(g.ntypes)
# # ['disease', 'drug']
# # 原始节点类型
# print(hg.ndata[dgl.NTYPE])
# # tensor([0, 0, 0, 1, 1, 1])
# # 原始的特定类型节点ID
# print(hg.ndata[dgl.NID])
# # tensor([0, 1, 2, 0, 1, 2])

# # 异构图中边类型的顺序
# print(g.etypes)
# # ['interacts', 'treats']
# # 原始边类型
# print(hg.edata[dgl.ETYPE])
# # tensor([0, 0, 1])
# # 原始的特定类型边ID
# print(hg.edata[dgl.EID])
# # tensor([0, 1, 0])