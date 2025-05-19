import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, residual=True,num_nodes=25):
        super(GCN, self).__init__()
        self.adaptive = adaptive
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes  # Store num_nodes

        # 线性变换：输入通道 -> 输出通道
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

        # 残差连接
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        if adaptive:
            # 自适应图 PA
            self.PA = nn.Parameter(torch.eye(num_nodes, dtype=torch.float32))
            print(f"GCN (MetaNet): Initialized adaptive PA with shape ({num_nodes}, {num_nodes})")
        else:
            # 固定图 A (如果提供了)
            # Store A, it will be used in forward if not adaptive
            self.A = None
            if A is not None:
                if isinstance(A, np.ndarray):
                    A_tensor = torch.from_numpy(A.astype(np.float32))
                elif isinstance(A, torch.Tensor):
                    A_tensor = A.float()
                else:
                    raise TypeError("Provided A must be a NumPy array or Torch Tensor")
                self.A = nn.Parameter(A_tensor, requires_grad=False)  # 固定矩阵
                print(f"GCN (MetaNet): Initialized fixed A from provided input. Shape: {self.A.shape}")
            else:
                # 如果 A 是 None 但又不是自适应，使用单位阵作为默认
                print("Warning: GCN (MetaNet) adaptive=False but A is None. Using identity matrix.")
                self.A = nn.Parameter(torch.eye(num_nodes, dtype=torch.float32), requires_grad=False)

        # 其他层定义
        self.alpha = nn.Parameter(torch.zeros(1))  # 缩放参数
        self.bn = nn.BatchNorm2d(out_channels)  # 批归一化
        self.soft = nn.Softmax(-2)  # softmax
        self.relu = nn.ReLU(inplace=True)  # relu

    def forward(self, x):
        y = None
        if self.adaptive:  # 可学习的邻接矩阵
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())  # 固定的邻接矩阵
        for i in range(self.num_subset):  # 对每个子集进行卷积操作
            z = self.convs[i](x, A[i], self.alpha)  # 对子集具体操作
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y

# --- 1. 元网络 (Graph Generator) ---
class MetaNetwork(nn.Module):
    def __init__(self,inmeta_channels, outmeta_channels, num_nodes, num_classes,
                 graph_init_type='identity', graph_args=None):
        """
        Args:
            input_dim (int): 输入特征的维度 (例如 C * V，如果将第一帧展平)
            num_joints (int): 关节点数量 (V)
        """
        super().__init__()

        self.data_bn = nn.BatchNorm1d(num_nodes * inmeta_channels * num_nodes)

        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.inmeta_channels = inmeta_channels  # 保存一下可能有用
        self.outmeta_channels = outmeta_channels  # 保存一下可能有用
        self.initialize_graph()

        self.gcn = GCN(inmeta_channels, outmeta_channels, A=None, adaptive=True, residual=True,num_nodes=self.num_nodes)


    def forward(self, x):
        N,C,T,V,M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data.bn(x)
        x = x.view(N, M, V, C, T).permute(0, 4, 1, 3, 2).contiguous().view(N*M, C , V, T)
        x = self.gcn(x)
        return x

    def initialize_graph(self):
        if self.graph_init_type == 'identity':
            adj = torch.eye(self.num_nodes, dtype=torch.float32)
        elif self.graph_init_type == 'uniform':
            adj = torch.ones(self.num_nodes, self.num_nodes, dtype=torch.float32) / self.num_nodes
        elif self.graph_init_type == 'skeleton':
            # 这里需要实现加载骨架图的逻辑，可能依赖 graph_args
            print(f"Warning: 'skeleton' graph_init_type not fully implemented in MetaNetwork. Defaulting to identity.")
            adj = torch.eye(self.num_nodes, dtype=torch.float32)
            # Example: adj = self.load_skeleton_graph(self.graph_args)
        else:
            raise ValueError(f"Unknown graph_init_type: {self.graph_init_type}")

        # 将图注册为 buffer。Buffer 是模型状态的一部分，会被保存和加载，
        # 但不会被优化器视为参数。这适合生成的图。
        self.register_buffer('generated_adj', adj)



