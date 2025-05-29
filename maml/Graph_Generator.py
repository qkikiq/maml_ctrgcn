import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import  torch
from    torch import nn
# 在文件顶部添加这行导入
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#
# class GCN(nn.Module):
#     def __init__(self, in_channels, out_channels, A, residual=True,num_nodes=25):
#         super(GCN, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_nodes = num_nodes  # Store num_nodes
#
#         # 线性变换：输入通道 -> 输出通道
#         for i in range(self.num_subset):
#             self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
#
#         # 残差连接
#         if residual:
#             if in_channels != out_channels:
#                 self.down = nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels, 1),
#                     nn.BatchNorm2d(out_channels)
#                 )
#             else:
#                 self.down = lambda x: x
#         else:
#             self.down = lambda x: 0
#
#         # 其他层定义
#         self.alpha = nn.Parameter(torch.zeros(1))  # 缩放参数
#         self.bn = nn.BatchNorm2d(out_channels)  # 批归一化
#         self.soft = nn.Softmax(-2)  # softmax
#         self.relu = nn.ReLU(inplace=True)  # relu
#
#     def forward(self, x, A):
#         """
#                 :param x: 输入特征，形状 (N, C, T, V)
#                 :param A: 输入邻接矩阵，形状 (K=3, V, V)
#                 """
#         N, C, T, V, M = x.size()
#
#         x = x.permute(0, 4, 3, 1, 2).contiguous().view( N, M * V * C, T)
#         x = self.data_bn(x)
#         x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view( N * M, C, T, V)
#         y = None
#
#         for i in range(self.num_subset):  # 对每个子集进行卷积操作
#             A_i = A[i] + self.alpha * torch.eye(self.num_nodes, device=x.device)
#             x_i = torch.einsum('nctv,vw->nctw', x, A_i)  # 图卷积
#             z = self.convs[i](x_i)
#             y = z if y is None else y + z
#
#         y = self.bn(y)
#         y += self.down(x)
#         y = self.relu(y)
#
#         return y


class net(nn.Module):
    def __init__(self, num_person=2, in_channels=3, num_nodes=25, dropout_rate=0.5,
                  mlp_hidden_dim=128, n_way=5, mlp_output_dim=256, lda_output_dim=128):
        super(net, self).__init__()

        self.V = num_nodes
        self.M = num_person
        self.C = in_channels

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)
        self.mlp = nn.Sequential(
            nn.Linear(num_nodes * in_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_output_dim) )
        


        #可学习的lda
        self.learnable_lda_layer = nn.Linear(mlp_output_dim, lda_output_dim)
        # 分类头，将LDA输出映射到类别得分
        self.classifier = nn.Linear(lda_output_dim, n_way)
        


    def forward(self, x, y): # x: [N,C,T,V,M]
        N ,C ,T ,V ,M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = x.permute(0, 2, 1, 3).contiguous().view(N * M, T, C * V)
        x = x.contiguous().view(N * M * T, C * V)
        x = self.mlp(x)   # [N*M*T, 256]
        c_new = x.size(1)
        # x = x.view(N * M, T, c_new)
        # x = x.view(N * M * T, c_new)

        # # LDA层：将特征投影到LDA空间
        # x_lda_out = self.learnable_lda_layer(x)
        # x_aggregated = x_lda_out.view(N * M, T, -1)
        # x_aggregated = x_aggregated.mean(dim=1)  #不合理
        # x_lda = self.classifier(x_aggregated)
        
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, device=x.device)
        else:
            y = y

        # 验证和准备实例标签
        if y.ndim == 1 and y.shape[0] == N: # 输入y的形状为(N,) - 样本级标签
            y = y.repeat_interleave(M) # Shape becomes (N*M,)
        elif y.ndim == 1 and y.shape[0] == N * M: # Input y already has shape (N*M,)
            y = y
        elif y.ndim == 2 and y.shape[0] == N and y.shape[1] == M: # Input y has shape (N,M)
            y = y.contiguous().view(N * M)
        else:
            raise ValueError(
                f"Input y (labels) has shape {y.shape}. Expected sample-level labels of shape ({N},), "
                f"or instance-level labels of shape ({N*M},) or ({N},{M}). "
                f"N={N}, M={M}."
            )
        
        y = y.long() # 确保标签为整数类型，LDA需要整数标签

        # # 扩展实例标签为帧标签，用于LDA: (N*M*T)
        lda_y = y.repeat_interleave(T) # 每个实例标签重复T次，对应每个时间帧

        #准备数据用于sklearn LDA：从计算图分离，移至CPU，转为NumPy数组
        lda_x = x.detach().cpu().numpy() # 特征数据准备
        lda_y = lda_y.cpu().numpy() # 标签数据准备

        # 初始化LDA
        lda = LinearDiscriminantAnalysis(solver='svd', n_components=None)

        # Fit LDA model and transform the data
        lda.fit(lda_x, lda_y)  # 拟合LDA模型并转换数据
        x_transform = lda.transform(lda_x)  # 将特征投影到LDA空间

        # 将转换后的数据转回PyTorch张量，并移回原始设备
        x_lda = torch.from_numpy(x_transform).to(x.device).float()
        
        # x_lda shape: (N*M*T, k-1) or (N*M*T, min(k-1, n_features))
        return x_lda

