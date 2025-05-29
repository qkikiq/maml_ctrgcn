import torch

data = torch.load('/dadaY/xinyu/pycharmproject/CTR-GCN-main/visal/learned_adj-epoch17-step400.pt')
print(type(data))
print(data.shape)      # 查看张量形状（4，75，25，128）
print(data.dtype)      # 数据类型
print(data)            # 打印全部数据（注意：如果太大，可能不适合全部打印）

