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

    #多分支卷积网络结构中权重的初始化
def conv_branch_init(conv, branches):
    weight = conv.weight   # 获取卷积层的权重参数
    n = weight.size(0)   # 获取输出通道数
    k1 = weight.size(1)   # 获取输入通道数
    k2 = weight.size(2)  # 获取卷积核的大小（假设为1D卷积或者是卷积核高度）
    # 使用正态分布初始化权重，均值为0，标准差根据He初始化的变体计算
    # 标准差考虑了分支数量，以防止多分支结构中的梯度爆炸或消失
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    # 如果卷积层有偏置参数，则将其初始化为0
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

    #用于初始化卷积层的参数
def conv_init(conv):
    # 检查卷积层是否有权重参数，如果有则进行初始化
    if conv.weight is not None:
        # 使用Kaiming初始化方法（又称He初始化）初始化权重
        # mode='fan_out'表示保持前向传播时方差一致，适合使用ReLU激活函数的网络
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
        # 检查卷积层是否有偏置参数，如果有则初始化为0
    if conv.bias is not None:
        # 检查卷积层是否有偏置参数，如果有则初始化为0
        nn.init.constant_(conv.bias, 0)

    #初始化批归一化(BatchNorm)层的参数
def bn_init(bn, scale):
    # 使用常数初始化BatchNorm层的权重(gamma)参数，值为传入的scale参数
    nn.init.constant_(bn.weight, scale)
    # 将BatchNorm层的偏置(beta)参数初始化为0
    nn.init.constant_(bn.bias, 0)


class Model(nn.Module):
    def __init__(self, in_channels, n_way, num_point, num_person, graph=None, graph_args=dict(),
                 max_N=500, drop_out=0, num_set=3, residual=True):
        super(Model, self).__init__()
        self.in_c = in_channels
        self.max_N = max_N
        self.n_way = n_way
        self.num_point = num_point
        self.num_person = num_person
        self.residual = residual
        self.drop_out_rate = drop_out
        self.num_set = num_set

        # 构建网络参数
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        # 导入图结构
        if graph is None:
            raise ValueError("图结构未指定")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        # 创建基础邻接矩阵
        A_base = self.graph.A
        self.register_buffer('A_base', torch.from_numpy(A_base.astype(np.float32)))

        # 如果需要动态邻接矩阵，则创建一个可学习的参数
        # self.A_learn = nn.Parameter(torch.zeros(num_set, num_point, num_point))
        # nn.init.xavier_uniform_(self.A_learn, gain=0.01)

        # A_learn = torch.zeros(num_set, num_point, num_point)
        # nn.init.xavier_uniform_(A_learn, gain=0.01)
        # self.vars.append(nn.Parameter(A_learn))
        # # 记录邻接矩阵在vars中的索引位置
        # self.A_learn_idx = len(self.vars) - 1

        # 定义网络配置
        self.config = []
        out_channels = 128
        self.num_subset = A_base.shape[0]

        # 数据归一化层
        self.config.append(('bn1d', [num_person * in_channels * num_point]))
        self.vars.append(nn.Parameter(torch.ones(num_person * in_channels * num_point)))
        self.vars.append(nn.Parameter(torch.zeros(num_person * in_channels * num_point)))
        running_mean = nn.Parameter(torch.zeros(num_person * in_channels * num_point), requires_grad=False)
        running_var = nn.Parameter(torch.ones(num_person * in_channels * num_point), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # 图卷积层参数
        for i in range(self.num_subset):
            self.config.append(('conv2d', [out_channels, in_channels, 1, 1, 1, 0]))
            w = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(out_channels)))

        # 批归一化层参数
        self.config.append(('bn2d', [out_channels]))
        self.vars.append(nn.Parameter(torch.ones(out_channels)))
        self.vars.append(nn.Parameter(torch.zeros(out_channels)))
        running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # 激活函数配置
        self.config.append(('relu', [True]))

        # 残差连接参数
        if residual and in_channels != out_channels:
            self.config.append(('res_conv2d', [out_channels, in_channels, 1, 1, 1, 0]))
            w = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1))
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            self.vars.append(nn.Parameter(torch.zeros(out_channels)))

            self.config.append(('res_bn2d', [out_channels]))
            self.vars.append(nn.Parameter(torch.ones(out_channels)))
            self.vars.append(nn.Parameter(torch.zeros(out_channels)))
            running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)
            self.vars_bn.extend([running_mean, running_var])

        # 全连接层参数
        # 确保 dropout 值在有效范围内
        valid_dropout = min(1.0, max(0.0, self.drop_out_rate))
        self.config.append(('dropout', [valid_dropout]))

        self.config.append(('linear', [n_way, out_channels]))
        w = nn.Parameter(torch.ones(n_way, out_channels))
        torch.nn.init.normal_(w, 0, math.sqrt(2. / n_way))
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(n_way)))

        #可学习邻接矩阵
        A_learn = torch.zeros(num_set, num_point, num_point)
        nn.init.xavier_uniform_(A_learn, gain=0.01)
        self.vars.append(nn.Parameter(A_learn))
        # 记录邻接矩阵在vars中的索引位置
        self.A_learn_idx = len(self.vars) - 1


        # 初始化实际层，用于保持兼容性
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(out_channels, n_way)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        # 初始化实际层
        bn_init(self.data_bn, 1)
        bn_init(self.bn, 1e-6)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / n_way))

    def L2_norm(self, A):
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4
        A = A / A_norm
        return A

    def forward(self, x, vars=None, bn_training=True):
        # 如果没有提供外部参数，则使用模型自身参数
        if vars is None:
            vars = self.vars

        # 初始化参数索引计数器
        idx = 0
        bn_idx = 0
        # 获取输入维度
        N, C, T, V, M = x.size()
        # 重塑数据用于批归一化
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # 批归一化（初始数据处理）
        name, param = self.config[0]    #归一化bn1d  150
        w, b = vars[idx], vars[idx + 1]   # 获取归一化层的权重和偏置
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]  # 获取统计量
        x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
        idx += 2
        bn_idx += 2
        # 重塑数据用于图卷积
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # 获取可学习邻接矩阵参数 - 从vars列表中获取
        A_learn = vars[self.A_learn_idx]
        # 合并基础邻接矩阵和可学习部分
        A = self.A_base.cuda(x.get_device()) + A_learn
        # 应用规范化确保稳定性
        A = self.L2_norm(A)  # 使用已定义的L2_norm函数

        # 扩展邻接矩阵如果需要
        if len(A.shape) == 3:
            A = A.unsqueeze(0).expand(N*M, -1, -1, -1)

        graph_list = []  # 初始化图列表，用于存储每个子集的图。
        y = None
        # 执行图卷积操作
        for i in range(self.num_subset):
            # 获取对应子图的邻接矩阵
            A_i = A[:, i] if len(A.shape) == 4 else A[i]  # 获取对应子图的邻接矩阵
            # 获取卷积权重和偏置
            name, param = self.config[1 + i]  # conv2d
            w, b = vars[idx], vars[idx + 1]
            idx += 2
            graph_list.append(A_i)  # 将生成的图添加到图列表中。
            # 执行图卷积 f_out = σ(A·X·W)
            x_transformed = x.view(N * M, C * T, V)
            z = torch.matmul(x_transformed, A_i)
            z = z.view(N * M, C, T, V) # 消息传递: 特征与邻接矩阵相乘
            temp = F.conv2d(z, w, b) # 特征变换

            y = temp if y is None else y + temp


        # 批归一化-对图卷积结果进行归一化
        name, param = self.config[1 + self.num_subset] # bn2d
        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        y = F.batch_norm(y, running_mean, running_var, weight=w, bias=b, training=bn_training)
        idx += 2
        bn_idx += 2

        # ReLU激活
        name, param = self.config[2 + self.num_subset]
        y = F.relu(y, inplace=param[0])

        # 残差连接
        if self.residual:
            curr_idx = idx
            if self.in_c != y.size(1):
                # 如果通道数不匹配，使用1x1卷积调整
                name, param = self.config[3 + self.num_subset]  # res_conv2d
                if name == 'res_conv2d':
                    w, b = vars[idx], vars[idx + 1]
                    idx += 2
                    res = F.conv2d(x, w, b)

                    name, param = self.config[4 + self.num_subset]  # res_bn2d
                    w, b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    res = F.batch_norm(res, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    idx += 2
                    bn_idx += 2
            else:
                res = x
            y = y + res  # 特征加残差

        # 池化
        c_new = y.size(1)
        y = y.view(N, M, c_new, -1)
        y = y.mean(3).mean(1)

        # Dropout- 随机置零防止过拟合
        drop_idx = -3 if self.residual and self.in_c != c_new else -2  # 确定dropout配置的索引
        name, param = self.config[drop_idx]
        if name == 'dropout':
            # 确保 dropout 概率在有效范围内
            dropout_rate = min(1.0, max(0.0, param[0]))
            if dropout_rate > 0:
                y = F.dropout(y, p=dropout_rate, training=bn_training)

        # 全连接分类
        name, param = self.config[-1]
        w, b = vars[idx], vars[idx + 1]
        output = F.linear(y, w, b)
        graph = torch.stack(graph_list, 1)
        graph = graph.view(N, M, -1, V, V).mean(1).view(N, -1)

        return output, graph

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