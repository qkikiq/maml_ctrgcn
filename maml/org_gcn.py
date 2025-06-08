import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

#初始化卷积层权重
def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

#初始化偏置和BN层
def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, max_N=500):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.num_point = A.shape[1]
        self.max_N = max_N  # 最大批次大小

        #固定的邻接矩阵 不参与训练
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        # 可训练的个性化邻接矩阵，初始为零
        self.A_personalized = nn.Parameter(
            torch.zeros(max_N, self.num_subset, self.num_point, self.num_point).float()
        )
        # 使用较小的常数初始化，避免一开始对原始邻接矩阵的过度影响
        nn.init.constant_(self.A_personalized, 1e-6)


        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))


        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x, vars=None,vars_bn=None):
        N, C, T, V = x.size()

        # 确保批次大小不超过预设的最大值
        batch_size = min(N, self.max_N)
        #固定邻接矩阵
        base_A = self.A.cuda(x.get_device())  # 3，25，25
        A = base_A.unsqueeze(0).expand(batch_size, -1, -1, -1)  # 扩展 A 的维度以匹配批次大小（50，3，25，25）
        # 个性化邻接矩阵
        A_personalized = torch.tanh(self.A_personalized[:batch_size])  # [N, num_subset, num_point, num_point]
        A_final = A + A_personalized  # 合成邻接矩阵 [N, ks, V, V]

        y = None  # 初始化输出特征为 None

        for i in range(self.num_subset):
            A_num = A_final[:, i]    # [batch_size, V, V]
            batch_results = []
            for j in range(batch_size):
                A1 = A_num[j]     # [V, V]
                A2 = x[j].view(C * T, V)  # [C * T, V]

                # 使用 F.conv2d 显式传递权重和偏置
                weight = vars[2 * i + 1] if vars else self.conv_d[i].weight
                bias = vars[2 * i + 2] if vars else self.conv_d[i].bias

                z = F.conv2d(
                    torch.matmul(A2, A1).view(1, C, T, V),
                    weight=weight,
                    bias=bias
                )
                batch_results.append(z)
                # 将所有样本结果堆叠成一个批次
            z = torch.cat(batch_results, dim=0)  # [N, C, T, V]
            y = z + y if y is not None else z
            # 如果批次大小超过最大值，处理剩余样本（使用默认邻接矩阵）

            # 批归一化参数
        bn_weight = vars_bn[2] if vars else self.bn.weight  # 假设批归一化权重存储在最后
        bn_bias = vars_bn[3] if vars else self.bn.bias
        y = F.batch_norm(y, self.bn.running_mean, self.bn.running_var, bn_weight, bn_bias, training=True)

        y = self.relu(y)

        return A_personalized[:batch_size], y


class Model(nn.Module):
    def __init__(self, n_way=5, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, max_N=100, num_set=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.n_way = n_way
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = unit_gcn(3, 128, A, max_N=max_N)

        self.fc = nn.Linear(128, n_way)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / n_way))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        # 将所有需要优化的参数存储在 nn.ParameterList 中
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()  #储存批归一化
        # 将 data_bn 的参数存储到 vars_bn
        self.vars_bn.extend([self.data_bn.weight, self.data_bn.bias])


        for name, param in self.l1.named_parameters():
            if isinstance(dict(self.l1.named_modules()).get('.'.join(name.split('.')[:-1]), None), nn.BatchNorm2d):
                self.vars_bn.append(param)
            else:
                self.vars.append(param)
        # 将 fc 的参数存储到 vars
        self.vars.extend([self.fc.weight, self.fc.bias])

    def forward(self, x, vars=None):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)   #vars_bn[0]
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        graph, x = self.l1(x, vars, self.vars_bn)  #graph:N,C,V,V  x:N,C,T,V

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        fc_weight = vars[7] if vars else self.fc.weight
        fc_bias = vars[8] if vars else self.fc.bias
        return F.linear(x, fc_weight, fc_bias), graph

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        因为初始参数会以生成器的形式返回，所以请覆盖/修改此函数
        :return:
        """
        return self.vars