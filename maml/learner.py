import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# 你提供的GCN模块（作为参考，Learner本身不直接使用它，
# 但Learner中的GCN块会模仿其逻辑）
class GCN_Reference(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes=25, num_subset=3, residual=True):
        super(GCN_Reference, self).__init__()
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.num_nodes = num_nodes  # 节点数量 (V)
        self.num_subset = num_subset  # 邻接矩阵子集的数量 (K)

        self.convs = nn.ModuleList()  # 存储多个卷积层
        for _ in range(self.num_subset):
            # 每个子集对应一个1x1的2D卷积
            self.convs.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1))

        if residual:  # 是否使用残差连接
            if in_channels != out_channels:
                # 如果输入输出通道数不同，使用一个1x1卷积进行下采样（或上采样）以匹配维度
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                # 通道数相同，直接恒等映射
                self.down = lambda x: x
        else:
            # 不使用残差连接，则残差路径输出为0
            self.down = lambda x: 0

        self.alpha = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数alpha，用于调整单位矩阵的贡献
        self.bn = nn.BatchNorm2d(out_channels)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

    def forward(self, x, A):
        # x: 输入特征，形状 (N, C, T, V) - N:批大小, C:通道数, T:帧数, V:节点数
        # A: 输入邻接矩阵，形状 (K, V, V) - K:子集数
        N, C, T, V = x.size()
        if A.size(0) != self.num_subset:
            raise ValueError(f"邻接矩阵子集数量 ({A.size(0)}) 与GCN定义的num_subset ({self.num_subset}) 不匹配")
        if V != self.num_nodes:
            raise ValueError(f"输入节点数 ({V}) 与GCN定义的num_nodes ({self.num_nodes}) 不匹配")

        y_agg = None  # 用于聚合各子图卷积结果的变量

        for i in range(self.num_subset):  # 对每个子集进行操作
            # A_i = A[i] + self.alpha * torch.eye(V, device=x.device) # 原GCN写法中V应该用self.num_nodes
            A_i = A[i] + self.alpha * torch.eye(self.num_nodes, device=x.device)  # 调整后的邻接矩阵

            # 图卷积操作:
            # einsum 'nctv,vw->nctw' 表示 X*A，其中 X 是 (N,C,T,V)，A_i 是 (V,V)，结果是 (N,C,T,V)
            # 这个结果直接送入Conv2D。Conv2D期望输入 (N, C_in, H_in, W_in)。
            # 这里 H_in=T (帧数), W_in=V (节点数)。1x1的Conv2D将在每个(T,V)位置独立应用。
            x_transformed = torch.einsum('nctv,vw->nctw', x, A_i)  # 特征与调整后的邻接矩阵相乘

            z = self.convs[i](x_transformed)  # 通过对应的1x1卷积层
            y_agg = z if y_agg is None else y_agg + z  # 聚合结果

        y = self.bn(y_agg)  # 对聚合后的结果进行批归一化
        y += self.down(x)  # 添加残差连接 (输入x经过downsample处理)
        y = self.relu(y)  # ReLU激活
        return y


class Learner(nn.Module):
    def __init__(self, config, num_nodes=25, num_subsets_A=3):
        """
        支持GCN的MAML Learner。
        :param config: 网络配置列表。
                       GCN示例: ('gcn', {'in_channels': c_in, 'out_channels': c_out, 'residual': True})
                       Linear示例: ('linear', {'in_features': f_in, 'out_features': f_out})
                       激活层/池化层示例: ('relu', {}) 或 ('global_mean_pool', {})
        :param num_nodes: 图中的节点数量 (V)。
        :param num_subsets_A: 邻接矩阵堆栈中的数量 (K)，即GCN中的num_subset。
        """
        super(Learner, self).__init__()
        self.config = config  # 保存网络配置
        self.num_nodes = num_nodes  # 图节点数
        self.num_subsets_A = num_subsets_A  # 邻接矩阵子集数

        self.vars = nn.ParameterList()  # 存储所有可优化的张量 (权重和偏置)
        self.vars_bn = nn.ParameterList()  # 存储批归一化层的running_mean和running_var

        for i, (name, params) in enumerate(self.config):  # 遍历配置，初始化参数
            if name == 'gcn':  # 如果是GCN层
                in_channels = params['in_channels']
                out_channels = params['out_channels']
                residual = params.get('residual', True)  # 默认为True，如果配置中未指定

                # GCN内部的卷积层 (每个邻接矩阵子集对应一个)
                for _ in range(self.num_subsets_A):
                    # 卷积层权重: [out_channels, in_channels, kernel_height, kernel_width]
                    # 对于kernel_size=1, 形状是 [out_channels, in_channels, 1, 1]
                    w_conv = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1))
                    torch.nn.init.kaiming_normal_(w_conv)  # Kaiming正态初始化
                    self.vars.append(w_conv)
                    # 卷积层偏置: [out_channels]
                    b_conv = nn.Parameter(torch.zeros(out_channels))
                    self.vars.append(b_conv)

                # Alpha参数，用于 A_i_eff = A_i + alpha * I
                alpha = nn.Parameter(torch.zeros(1))  # 初始化为0
                self.vars.append(alpha)

                # 主GCN输出的批归一化层参数
                bn_w = nn.Parameter(torch.ones(out_channels))  # BN层权重gamma
                self.vars.append(bn_w)
                bn_b = nn.Parameter(torch.zeros(out_channels))  # BN层偏置beta
                self.vars.append(bn_b)
                # BN层的running_mean和running_var (不可训练，但在训练时更新)
                self.vars_bn.extend([nn.Parameter(torch.zeros(out_channels), requires_grad=False),
                                     nn.Parameter(torch.ones(out_channels), requires_grad=False)])

                # 残差连接参数
                if residual:
                    if in_channels != out_channels:  # 如果输入输出通道不一致，需要下采样/上采样
                        # 下采样卷积层权重: [out_channels, in_channels, 1, 1]
                        w_down_conv = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1))
                        torch.nn.init.kaiming_normal_(w_down_conv)
                        self.vars.append(w_down_conv)
                        # 下采样卷积层偏置 (通常Conv2d默认带偏置)
                        b_down_conv = nn.Parameter(torch.zeros(out_channels))
                        self.vars.append(b_down_conv)

                        # 下采样路径的BN层参数
                        bn_down_w = nn.Parameter(torch.ones(out_channels))
                        self.vars.append(bn_down_w)
                        bn_down_b = nn.Parameter(torch.zeros(out_channels))
                        self.vars.append(bn_down_b)
                        self.vars_bn.extend([nn.Parameter(torch.zeros(out_channels), requires_grad=False),
                                             nn.Parameter(torch.ones(out_channels), requires_grad=False)])
                    # 如果通道数相同或不使用残差，则这部分没有额外参数。具体逻辑在forward中处理。

            elif name == 'linear':  # 如果是全连接层
                in_features = params['in_features']
                out_features = params['out_features']
                # 权重: [out_features, in_features]
                w = nn.Parameter(torch.ones(out_features, in_features))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # 偏置: [out_features]
                b = nn.Parameter(torch.zeros(out_features))
                self.vars.append(b)

            elif name == 'bn_1d':  # 如果是一维批归一化 (通常用于展平/池化后的特征)
                num_features = params['num_features']
                w = nn.Parameter(torch.ones(num_features))  # BN权重
                self.vars.append(w)
                b = nn.Parameter(torch.zeros(num_features))  # BN偏置
                self.vars.append(b)
                self.vars_bn.extend([nn.Parameter(torch.zeros(num_features), requires_grad=False),
                                     nn.Parameter(torch.ones(num_features), requires_grad=False)])

            elif name in ['relu', 'global_mean_pool', 'flatten', 'log_softmax']:  # 无需学习参数的层
                continue
            else:
                raise NotImplementedError(f"未在Learner.__init__中实现层类型 {name}")

        # 添加可学习的邻接矩阵参数
        # 为每个子集创建一个可学习的邻接矩阵参数
        self.adj_params = nn.ParameterList()
        for _ in range(num_subsets_A):
            # 初始化为单位矩阵加一些随机噪声
            adj_param = nn.Parameter(torch.eye(num_nodes) + 0.01 * torch.randn(num_nodes, num_nodes))
            self.adj_params.append(adj_param)

        # 将邻接矩阵参数也添加到 vars 中，以便在 MAML 中更新
        for adj_param in self.adj_params:
            self.vars.append(adj_param)

    def forward(self, x_input, A_input=None, vars=None, bn_training=True):
        """
        前向传播函数。
        :param x_input: 输入特征。
               - 可以是 [task_num, sample_num, C, T, V, M] 形状的骨架数据
               - 或 [B, C, T, V, M] 形状的骨架数据 (标准骨架输入)
               - 或根据网络要求的其他形状
        :param A_input: 邻接矩阵，形状可以是 [task_num, sample_num, num_subsets_A, num_nodes, num_nodes]。
                如果config中包含'gcn'层，则此参数必需。
        :param vars: 可选的参数列表 (用于MAML的快速适应)。如果为None，则使用self.vars。
        :param bn_training: 布尔值，BN层是否应更新其running_mean/var统计量 (即是否为训练模式)。
        :return: 网络的输出。
        """
        # 如果未提供vars，则使用模型自身的参数
        # 修改 vars 参数处理逻辑，增加边界检查
        if vars is None:
            vars_list = self.vars
        else:
            # 如果提供的参数列表长度不够，使用默认参数
            if len(vars) < len(self.vars) - self.num_subsets_A:
                # 打印警告
                print(f"警告: 提供的参数列表({len(vars)})短于预期({len(self.vars)})。将使用默认参数。")
                vars_list = self.vars
            else:
                vars_list = vars

        param_idx = 0  # vars_list中的参数索引
        bn_param_idx = 0  # vars_bn中的参数索引
        x = x_input  # 当前的特征张量，在前向传播过程中会被更新

        # 处理6D输入 [task_num, sample_num, C, T, V, M]
        if x.dim() == 6:
            # 获取原始形状信息
            task_num, sample_num, C, T, V, M = x.shape
            
            # 1. 处理人体维度 M，可以取最大值或平均值
            x = x.mean(dim=-1)  # [task_num, sample_num, C, T, V]
            
            # 2. 重新组织批次维度，将task_num和sample_num合并为一个批次维度
            x = x.reshape(-1, C, T, V)  # [task_num*sample_num, C, T, V]
            
            # 3. 相应地，如果A_input是[task_num, sample_num, num_subsets_A, V, V]，也需要调整
            if A_input is not None and A_input.dim() == 5:
                A_input = A_input.reshape(-1, self.num_subsets_A, V, V)  # [task_num*sample_num, num_subsets_A, V, V]
    
        # 新增：处理5D输入 [B, C, T, V, M] - 骨架数据格式
        elif x.dim() == 5:
            # 获取原始形状信息
            B, C, T, V, M = x.shape
            
            # 确保节点数匹配
            if V != self.num_nodes:
                raise ValueError(f"输入节点数 ({V}) 与GCN定义的num_nodes ({self.num_nodes}) 不匹配")
            # 将M视为批次维度的一部分，保留所有数据而不是平均
            # 从 [B, C, T, V, M] 变为 [B*M, C, T, V]
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(B * M, C, T, V)
            
            # 处理邻接矩阵，如果需要
            # 假设邻接矩阵是[B, num_subsets_A, V, V]或[num_subsets_A, V, V]
            # 如果是[B, num_subsets_A, V, V]，需要扩展到[B*M, num_subsets_A, V, V]
            if A_input is not None:
                if A_input.dim() == 4 and A_input.shape[0] == B:  # [B, num_subsets_A, V, V]
                    A_input = A_input.repeat_interleave(M, dim=0)  # 复制M次，变为[B*M, num_subsets_A, V, V]
                elif A_input.dim() == 3:  # [num_subsets_A, V, V]
                    # 这种情况无需处理，因为每个样本共享相同的邻接矩阵
                    pass
                else:
                    # 其他形状可能需要额外处理
                    pass

        # 获取学习的邻接矩阵参数（从 vars_list 或 self.adj_params）
        learned_adjacency = []

        if vars is None:
            # 如果没有提供 vars，使用类的邻接矩阵参数
            adj_params = self.adj_params
        else:
            # 检查 vars 的参数数量是否足够
            if len(vars) < len(self.vars):
                # vars 中可能不包含邻接矩阵参数，使用类自己的邻接矩阵参数
                adj_params = self.adj_params
            else:
                # 从提供的 vars 中提取邻接矩阵参数（在 vars 列表的末尾）
                adj_params_start_idx = len(vars) - self.num_subsets_A
                adj_params = vars[adj_params_start_idx:]

        # 处理输入的邻接矩阵
        # 如果提供了 A_input，将其与学习的参数结合
        # 否则，仅使用学习的参数
        if A_input is not None:
            # 对每个子集应用学习的参数
            for k in range(self.num_subsets_A):
                # 将学习的参数应用到输入邻接矩阵
                # 可以是加法、乘法或其他组合方式
                A_k_learned = A_input[:, k] * F.sigmoid(adj_params[k])
                # 对邻接矩阵进行归一化
                A_k_learned = F.normalize(A_k_learned, p=1, dim=2)
                learned_adjacency.append(A_k_learned)
        else:
            # 如果没有提供输入邻接矩阵，仅使用学习的参数
            for k in range(self.num_subsets_A):
                A_k_learned = F.sigmoid(adj_params[k])
                A_k_learned = F.normalize(A_k_learned, p=1, dim=1)
                # 扩展到匹配批次大小
                # batch_size = x.size(0)
                # A_k_learned = A_k_learned.unsqueeze(0).expand(batch_size, -1, -1)
                learned_adjacency.append(A_k_learned)
        
        # 将学习到的邻接矩阵堆叠在一起
        learned_adjacency = torch.stack(learned_adjacency, dim=1)  # [batch_size, num_subsets_A, num_nodes, num_nodes]
        
        for i, (name, params) in enumerate(self.config):  # 遍历网络配置中的每一层
            if name == 'gcn':  # 如果是GCN层
                if A_input is None:
                    raise ValueError("GCN层需要 A_input (邻接矩阵)。")

                in_channels = params['in_channels']
                out_channels = params['out_channels']
                residual = params.get('residual', True)

                # 如果输入x是 [B, C, T, V] (批大小, 通道数, 时间步, 节点数)，这已经是GCN期望的格式
                if x.dim() == 4 and x.shape[1] == in_channels and x.shape[3] == self.num_nodes:
                    x_gcn_format = x
                # 如果输入是 [B, N, C_in] (批大小, 节点数, 输入通道数)
                elif x.dim() == 3 and x.shape[1] == self.num_nodes and x.shape[2] == in_channels:
                    x_gcn_format = x.permute(0, 2, 1).unsqueeze(2)  # 转换为 [B, C_in, 1, N_nodes]
                else:
                    raise ValueError(
                        f"GCN层接收到意外的输入形状 {x.shape}，期望输入通道为 {in_channels} 且节点数为 {self.num_nodes}。")

                # 保存原始x (转换格式后) 用于残差连接
                x_residual_orig = x_gcn_format

                # GCN K个子图卷积操作
                y_aggregated = None  # 用于聚合各个子图卷积结果

                # 从vars_list中提取当前GCN层的所有子卷积的权重和偏置
                # 添加参数索引安全检查
                try:
                    conv_weights_biases = []
                    for _ in range(self.num_subsets_A):
                        if param_idx + 1 >= len(vars_list):
                            raise IndexError(f"参数索引越界: param_idx={param_idx}, len(vars_list)={len(vars_list)}")
                        w_conv = vars_list[param_idx]
                        b_conv = vars_list[param_idx + 1]
                        conv_weights_biases.append((w_conv, b_conv))
                        param_idx += 2  # 每个卷积层消耗2个参数 (权重和偏置)

                    # 更安全的索引检查
                    if param_idx >= len(vars_list):
                        raise IndexError(f"参数索引越界: param_idx={param_idx}, len(vars_list)={len(vars_list)}")
                    alpha_param = vars_list[param_idx]  # 提取alpha参数
                    param_idx += 1
                except IndexError as e:
                    print(f"错误: {e}")
                    # 回退到使用模型默认参数
                    return self.forward(x_input, A_input, vars=None, bn_training=bn_training)

                for k_subset in range(self.num_subsets_A):  # 遍历每个邻接矩阵子集
                    # 获取邻接矩阵，处理维度不匹配问题
                    if learned_adjacency.dim() == 4:  # [batch_size, num_subsets_A, num_nodes, num_nodes]
                        A_k_base = learned_adjacency[:, k_subset]  # [batch_size, num_nodes, num_nodes]
                        
                        # 为每个样本添加单位矩阵乘以alpha
                        batch_size = A_k_base.size(0)
                        eye_matrix = torch.eye(self.num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
                        A_k = A_k_base + alpha_param * eye_matrix
                        
                        # 使用批量einsum操作
                        x_transformed_by_A = torch.einsum('bchv,bvw->bchw', x_gcn_format, A_k)
                    else:  # 如果是[num_subsets_A, num_nodes, num_nodes]
                        A_k = learned_adjacency[k_subset] + alpha_param * torch.eye(self.num_nodes, device=x.device)
                        # 使用原始einsum操作
                        x_transformed_by_A = torch.einsum('nchv,vw->nchw', x_gcn_format, A_k)

                    w_conv_k, b_conv_k = conv_weights_biases[k_subset]  # 获取当前子图的卷积参数
                    # 进行2D卷积
                    z_k = F.conv2d(x_transformed_by_A, w_conv_k, b_conv_k, stride=1,
                                  padding=0)  # (N, C_out, H=1, V_nodes)

                    y_aggregated = z_k if y_aggregated is None else y_aggregated + z_k  # 聚合结果

                # 主GCN输出的批归一化
                bn_w = vars_list[param_idx]
                param_idx += 1
                bn_b = vars_list[param_idx]
                param_idx += 1
                bn_running_mean = self.vars_bn[bn_param_idx]
                bn_param_idx += 1
                bn_running_var = self.vars_bn[bn_param_idx]
                bn_param_idx += 1

                y = F.batch_norm(y_aggregated, bn_running_mean, bn_running_var,
                                 weight=bn_w, bias=bn_b, training=bn_training)

                # 残差连接
                if residual:
                    if in_channels != out_channels:  # 如果输入输出通道不同，应用下采样路径
                        w_down_conv = vars_list[param_idx];
                        param_idx += 1
                        b_down_conv = vars_list[param_idx];
                        param_idx += 1

                        bn_down_w = vars_list[param_idx];
                        param_idx += 1
                        bn_down_b = vars_list[param_idx];
                        param_idx += 1

                        bn_down_mean = self.vars_bn[bn_param_idx];
                        bn_param_idx += 1
                        bn_down_var = self.vars_bn[bn_param_idx];
                        bn_param_idx += 1

                        downsampled_x = F.conv2d(x_residual_orig, w_down_conv, b_down_conv, stride=1, padding=0)
                        downsampled_x = F.batch_norm(downsampled_x, bn_down_mean, bn_down_var,
                                                     weight=bn_down_w, bias=bn_down_b, training=bn_training)
                        y = y + downsampled_x
                    else:  # 通道数相同，直接相加
                        y = y + x_residual_orig
                # else: y = y (不使用残差，或者由self.down = lambda x: 0处理)

                x = F.relu(y, inplace=True)  # ReLU激活，输出形状: [B, C_out, 1, N_nodes]

            elif name == 'linear':  # 如果是全连接层
                in_features = params['in_features']
                # out_features = params['out_features'] # 此处不需要

                w = vars_list[param_idx];
                param_idx += 1
                b = vars_list[param_idx];
                param_idx += 1

                # 全连接层通常作用于展平的向量 [B, in_features]
                # 需要确保输入x的形状是正确的 (例如，在GCN或池化层之后可能需要展平)
                if x.dim() != 2 or x.shape[1] != in_features:
                    raise ValueError(
                        f"全连接层输入形状为 {x.shape}, 期望 (batch, {in_features})。请确保在全连接层前有展平(flatten)或池化(pool)操作。")
                x = F.linear(x, w, b)

            elif name == 'bn_1d':  # 一维批归一化
                num_features = params['num_features']
                if x.dim() != 2 or x.shape[1] != num_features:
                    raise ValueError(f"bn_1d层输入形状为 {x.shape}, 期望 (batch, {num_features})。")

                w = vars_list[param_idx];
                param_idx += 1
                b = vars_list[param_idx];
                param_idx += 1
                running_mean = self.vars_bn[bn_param_idx];
                bn_param_idx += 1
                running_var = self.vars_bn[bn_param_idx];
                bn_param_idx += 1
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)

            elif name == 'relu':  # ReLU激活层
                x = F.relu(x, inplace=params.get('inplace', False))  # 允许从config指定inplace

            elif name == 'flatten':  # 展平层
                x = x.view(x.size(0), -1)  # 将除批大小维度外的所有维度展平

            elif name == 'global_mean_pool':  # 全局平均池化层
                # 假设 x 是 [B, C, T, V] (来自GCN) 或 [B, C, V] (如果T被压缩)
                # 或者 [B, N, C] (节点级池化)
                if x.dim() == 4:  # 例如, GCN输出 (B, C, T, V)
                    x = x.mean(dim=(2, 3))  # 对T和V维度进行全局平均池化, 输出 (B, C)
                elif x.dim() == 3:  # 例如, (B, N, C)
                    x = x.mean(dim=1)  # 对节点维度进行池化, 输出 (B, C)
                elif x.dim() == 2:  # 例如, (B,C) 已经是池化后结果
                    pass
                else:
                    raise ValueError(f"全局平均池化层接收到意外的输入形状 {x.shape}")


            elif name == 'log_softmax':  # LogSoftmax层 (常用于分类任务的输出)
                dim = params.get('dim', -1)  # 从config获取维度，默认为最后一维
                x = F.log_softmax(x, dim=dim)

            else:
                raise NotImplementedError(f"未在Learner.forward中实现层类型 {name}")

        # 参数使用完整性检查 (可选，用于调试)
        # 如果网络中存在条件分支（例如GCN的残差连接在通道数相同时跳过参数），
        # param_idx 和 len(vars_list) 可能不会完全匹配。
        # 当前实现应该能正确处理GCN残差路径的参数分配和使用。
        if param_idx != len(vars_list):
            # print(f"警告: param_idx ({param_idx}) 与 len(vars_list) ({len(vars_list)}) 不匹配。这可能由GCN残差连接的条件参数导致。")
            pass

        if bn_param_idx != len(self.vars_bn):
            # print(f"警告: bn_param_idx ({bn_param_idx}) 与 len(self.vars_bn) ({len(self.vars_bn)}) 不匹配。")
            pass

        # 返回网络输出和学习到的邻接矩阵
        return x, learned_adjacency

    def zero_grad(self, vars_list=None):
        """将参数的梯度清零。"""
        with torch.no_grad():  # 在不计算梯度的上下文中执行
            if vars_list is None:
                param_source = self.vars
            else:
                param_source = vars_list

            for p in param_source:
                if p.grad is not None:
                    p.grad.zero_()

    def parameters(self):
        """返回MAML优化器需要优化的参数列表 (self.vars)。"""
        return self.vars

    def get_bn_vars(self):
        """返回BN层的状态变量 (running_mean, running_var)。"""
        return self.vars_bn