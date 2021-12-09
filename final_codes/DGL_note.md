### **取边特征的时候，这是取0、3边的特征**
print(g.edata['x'][th.tensor([0, 3])] )  # 获取边0和3的特征

### **构建图的关键**
- 弄到边关系 包括源和目的 最好是两列
- 弄到节点特征  

有以上两个过程，只要能正确对应就没问题了。  
相比较而言HAR的图是比较好构建的，而SNS的图通常可能比较大，就不容易构建。对于节点属性通常要求是特征矩阵，这就要求在构建过程中不断把节点、边的各种属性转换成各自不同维度的等长特征向量。

HAR在构建图的时候别直接构建，还是从csv文件构建比较好。这样从通道往下查，传感器对应1、2、3、4...，根据传感器分布分配完节点号后回头构建图更为方便。就可以方便的构图了。

从外部文件构建图
[csv文件构建俱乐部图](https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/basic_tasks/1_load_data.ipynb)  
csv文件构建图的时候可以参考

dgl.save_graphs()  
dgl.load_graphs()  
数据处理完可以直接以图形式保存，如果处理速度快就无所谓了，如果慢的话可能就得一次处理完以图形式存储。

### __DGL GPU__
首先构建图对象，g.to("cuda:0")同样方式送到gpu上，特征也被转移到了gpu上
也可先将源和目标节点转换到GPU上，下一步构建的图自动在gpu上。

### __下一步还要考虑异质图构建__
异质图构建

DGL的异质图构建较为方便，在做HAR对比实验的时候可以采用异质图-》同质体对比的形式来看实验对比结果。  
而且在节点特征数据当中由于在采集的时候已经按照所有的传感器采样频率相同处理，因此更容易转换成同质图来进行处理。



### __消息传递 重点__

__1.__ 边权重是首先要计算的，有上一层的节点和边值来计算  
__2.__ 根据边权重来送入聚合函数，邻居节点的值、边值做汇聚  
__3.__ 聚合的邻居信息、自身的特征信息计算新的节点特征做更新。

假设节点 v 上的的特征为 xv∈Rd1，边 (u,v) 上的特征为 we∈Rd2。 消息传递范式 定义了以下逐节点和边上的计算：

边上计算: m(t+1)e=ϕ(x(t)v,x(t)u,w(t)e),(u,v,e)∈E.
点上计算: x(t+1)v=ψ(x(t)v,ρ({m(t+1)e:(u,v,e)∈E})).
在上面的等式中， ϕ 是定义在每条边上的消息函数，它通过将边上特征与其两端节点的特征相结合来生成消息。 聚合函数 ρ 会聚合节点接受到的消息。 更新函数 ψ 会结合聚合后的消息和节点本身的特征来更新节点的特征。

[原始公式链接](https://docs.dgl.ai/guide_cn/message.html)

### __消息传递 理解总结__
__1. 消息函数：__ 接收edges，为EdgeBatch实例，接收一批边，代表要根据此对象中的所有边逐次进行更新，更新的时候要用到源节点、目的节点、原有数据值，都可以直接从edges中直接提取。此步骤是以所有的边为核心计算对象。  
消息函数的计算对象是源、目的、原值，内置消息函数：dgl.function.u_add_v('hu', 'hv', 'he')
源：hu、目的：hv、新的边值：he
自定义消息函数：
def message_func(edges):
     return {'he': edges.src['hu'] + edges.dst['hv']}
__2. 聚合函数：__ 接收nodes，为NodeBatch实例，接收一批点，代表要根据此对象中的所有点逐次进行更新，更新的时候要用到邻居信息，由成员属性mailbox 存储收到的节点消息。聚合函数有sum、max、min等。
dgl.function.sum('m', 'h')
import torch
def reduce_func(nodes):
     return {'h': torch.sum(nodes.mailbox['m'], dim=1)}  
__3. 更新函数：__ 接收聚合函数里的nodes，对聚合函数结果进行操作，与本节点特征组合输出作为新的节点特征。


__4.逐边函数：__ 此函数在不涉及消息传递，单独调用逐边计算。默认情况下此函数会更新所有的边。
import dgl.function as fn
graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

__5.update_all函数：__ 此函数是高级API，包含了消息生成、消息聚合、节点特征更新所有操作。
```python
def updata_all_example(graph):
    # 在graph.ndata['ft']中存储结果
    graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                     fn.sum('m', 'ft')) #调用消息函数、聚合函数
    # 在update_all外调用更新函数
    final_ft = graph.ndata['ft'] * 2#调用了更新函数
    return final_ft
```

消息函数其实不仅仅是更新边的信息，而是生成消息，这个消息里可以包含源、目的、边的数值，只不过是以边的方向进行传递,同时体现在边的特征中。聚合函数负责把算到的结果聚合到节点上，最后节点可以用聚合之做更新操作。

__6.子图上进行消息传递:__

```python
nid = [0, 2, 3, 6, 7, 9] #子图节点号
sg = g.subgraph(nid)    #子图
sg.update_all(message_func, reduce_func, apply_node_func)#子图消息更新
```
子图上先更新一波，而不是在全图上更新。
这样消息更新分成多步，可以有效的定义不同部分的消息更新策略，控制消息更新的范围。

__7.边上进行消息更新__
```python
import dgl.function as fn

# 假定eweight是一个形状为(E, *)的张量，E是边的数量。
graph.edata['a'] = eweight
graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                 fn.sum('m', 'ft'))
```

__8.heterogeneous graph异质图__
 
 - 每个关系单独计算  
 - 所有关系聚合到节点上  

```python
import dgl.function as fn

for c_etype in G.canonical_etypes:
    srctype, etype, dsttype = c_etype
    Wh = self.weight[etype](feat_dict[srctype])
    # 把它存在图中用来做消息传递
    G.nodes[srctype].data['Wh_%s' % etype] = Wh
    # 指定每个关系的消息传递函数：(message_func, reduce_func).
    # 注意结果保存在同一个目标特征“h”，说明聚合是逐类进行的。
    funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
# 将每个类型消息聚合的结果相加。
G.multi_update_all(funcs, 'sum')
# 返回更新过的节点特征字典
return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
```

### __图神经网络forward函数  ~~这一章没通的有点多啊~~__
expand_as_pair函数对于同构图、异构图、二分图有不同的处理方式。这个位置没太看通。  
[expand_as_pair函数解析](https://docs.dgl.ai/guide_cn/nn-forward.html#id1)  
[异质图图卷积模块](https://docs.dgl.ai/guide_cn/nn-heterograph.html)  


### __数据下载与处理__
对zip 和gz文件有方便的方式
- 拼接文件保存路径
- 从指定连接（self.url）中下载数据文件，存储到文件保存路径中download函数
- .zip extract_archive()函数就可以进行解压
- .gz 文件解压方式可以参考[BitcoinOTCDataset的_extract_gz](https://docs.dgl.ai/_modules/dgl/data/bitcoinotc.html#BitcoinOTCDataset)


### __图神经网络HAR实验研究步骤__
  1. 数据转化成图数据 包括特征送入、图结构节点与边关系送入。（模仿着改，HAR图关系简单，用通道的顺序来做节点）
  2. 用基本的图神经网络跑一下
  3. 复现一下TCN+GCN的论文
  4. 自己提出网络结构，在这上边做其他的实验。