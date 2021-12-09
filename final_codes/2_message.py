# 暂时没有完整代码   测一个求和相加。

import torch

data = torch.randn(3,5)

result = torch.sum(data, dim=1)

print(result == data.sum(dim=1))