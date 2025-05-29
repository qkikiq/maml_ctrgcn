import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import torch
from torch import nn
# 在文件顶部添加这行导入
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True,num_nodes=25,num_person=2,drop_out=0,
                 n_way=5):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes  # Store num_nodes

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)

        self.convs = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

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

        # 其他层定义
        self.alpha = nn.Parameter(torch.zeros(1))  # 缩放参数
        self.bn = nn.BatchNorm2d(out_channels)  # 批归一化
        self.soft = nn.Softmax(-2)  # softmax
        self.relu = nn.ReLU(inplace=True)  # relu

        self.fc = nn.Linear(out_channels , n_way)

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, adj,vars=None, bn_training=True ):
        """
                :param x: 输入特征，形状 (N, C, T, V)
                :param A: 输入邻接矩阵，形状 (K=3, V, V)
                """
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view( N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view( N * M, C, T, V)

        # 添加内存清理
        torch.cuda.empty_cache()

        adj = adj + self.alpha * torch.eye(self.num_nodes, device=x.device)
        x = torch.einsum('nctv,nvw->nctw', x, adj)  # 图卷积
        z = self.convs(x)

        y = self.bn(z)
        y += self.down(x)
        y = self.relu(y)

        c_new = y.size(1)
        x = y.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)


class net(nn.Module):
    def __init__(self, num_person=2, in_channels=3, num_nodes=25, dropout_rate=0.5,
                 mlp_hidden_dim=128, n_way=5, mlp_output_dim=256, lda_output_dim=128):
        super(net, self).__init__()

        self.V = num_nodes
        self.M = num_person
        self.C = in_channels

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)

        # Head 1: MLP (N*M, C*V, T) -> (N*M*T, 256)
        self.mlp_head1 = nn.Sequential(
            nn.Linear(num_nodes * in_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_output_dim))

        # Head 2: 生成邻接矩阵 (N*M, V, V)
        self.mlp_head2 = nn.Sequential(
            nn.Linear(num_nodes * in_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, num_nodes*num_nodes))

        self.vars = nn.ParameterList()

    def forward(self, x, vars=None, bn_training=True):  # x: [N,C,T,V,M]
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        #每帧特征
        x_T = x.permute(0, 2, 1, 3).contiguous().view(N * M, T, C * V)
        x_T = x_T.contiguous().view(N * M * T, C * V)
        x_T = self.mlp_head1(x_T)  # [N*M*T, 256]
        # x_T = x_T.view(N * M, T, -1)

        #邻接矩阵
        window_size = 30
        stride = 30  # 或者更小，比如15
        num_windows = (T - window_size) // stride + 1
        adj_list = []
        for i in range(num_windows):
            t_start = i * stride
            t_end = t_start + window_size  # 窗口结束时间点
            # 切出当前时间窗口段: x_window shape = [N*M, C, window_size, V]
            x_win = x[:, :, t_start:t_end, :]  # [N*M, C, 30, V]
            # 时间维 mean pooling: [N*M, C, 30, V] → [N*M, C, V]
            x_win = x_win.mean(dim=2)  # 沿 T 聚合
            x_win = x_win.view(N * M, -1)   # 展平成向量： [N*M, C*V]
            adj_matrix = self.mlp_head2(x_win)   # MLP 预测邻接矩阵: [N*M, V*V]
            adj = adj_matrix.view(N * M, V, V)
            adj = (adj + adj.transpose(1, 2)) / 2 # 对称 & 非负
            adj = torch.relu(adj)
            adj_list.append(adj)  # 收集当前窗口的邻接矩阵
            # 将所有窗口的邻接矩阵堆叠成 [N*M, num_windows, V, V]
            adj = torch.stack(adj_list, dim=1)
            adj = adj.mean(dim=1)  # [N*M, V, V]
        return x_T, adj


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        因为初始参数会以生成器的形式返回，所以请覆盖/修改此函数
        :return:
        """
        return self.vars
