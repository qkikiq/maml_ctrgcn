import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class GraphGenerator(nn.Module):
    def __init__(self,n_way=5 , num_nodes=25, num_person=2, in_channels=3):
        super(GraphGenerator, self).__init__()
        self.n_way = n_way
        self.V = num_nodes
        self.M = num_person
        self.C = in_channels

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)
        self.mlp = nn.Sequential(
            nn.Linear(num_nodes * in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_nodes * num_nodes) )




    def forward(self, x ): # x: [N,C,T,V,M]
        task_num, N,C,T,V,M = x.shape
        # 在时间维度上取平均，处理多人维度M
        x = x.permute(0, 1, 5, 4, 2, 3).contiguous().view(task_num * N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(task_num, N, M, V, C, T).permute(0, 1, 2, 4, 5, 3).contiguous().view(task_num * N * M, C, T, V)
        x,_ = torch.max(x, dim= 2)  # [N, C, V]
        # 展平特征
        x_flat = x.reshape(task_num*N*M, -1)  # [task_num, N, C*V]
        # 生成邻接矩阵
        A = self.mlp(x_flat).view(task_num, N, M, V, V)
        A = A.mean(dim = 2)  # 多人平均： [task_num, N, V, V]
        # 归一化边权重
        A = F.softmax(A, dim=-1)  # [N, V, V]

        # segment_len = 3
        # num_segments = T // segment_len
        # x = x.permute(0, 1, 5, 4, 2, 3).contiguous().view(task_num * N, M * V * C, T)
        # x = self.data_bn(x)
        # x = x.view(task_num, N, M, V, C, T).permute(0, 1, 2, 4, 5, 3).contiguous().view(task_num * N * M, C, T, V)
        # B = task_num * N * M
        #
        # # ======== 切片并聚合片段特征 =========
        # x = x[:, :, :num_segments * segment_len, :]  # truncate if not divisible
        # x = x.view(B, C, num_segments, segment_len, V)  # [B, C, num_segments, 3, V]
        # x_seg = x.mean(dim=3)  # [B, C, num_segments, V]
        # x_seg = x_seg.permute(0, 2, 3, 1).contiguous()  # [B, num_segments, V, C]
        #
        # # ======== 对每段生成邻接矩阵 A =========
        # A_list = []
        # for i in range(num_segments):
        #     x_i = x_seg[:, i, :, :]  # [B, V, C]
        #     x_flat = x_i.reshape(B, -1)  # [B, V * C]
        #     A_i = self.mlp(x_flat)  # [B, V*V]
        #     A_i = A_i.view(B, V, V)
        #     A_i = F.softmax(A_i, dim=-1)
        #     A_list.append(A_i)
        #
        # A = torch.stack(A_list, dim=1)  # [B, num_segments, V, V]
        #
        # # ======== reshape回原始批次维度 ========
        # A = A.view(task_num, N, M, num_segments, V, V)
        # A = A.mean(dim=2)  # 多人平均： [task_num, N, num_segments, V, V]
        # A = A.view(task_num * N, num_segments, V, V)
        return A

class Learner(nn.Module):
    def __init__(self, num_nodes=25, in_channels=3, out_channels=128, num_classes=5):
        super(Learner, self).__init__()
        self.num_nodes = num_nodes
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(64)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 分类头
        self.fc = nn.Linear(out_channels * num_nodes, num_classes)

        self.vars = nn.ParameterList()  # 存储网络中所有的参数（权重、偏置）、并且支持优化

    def forward(self, x, A, vars=None):  # 添加vars参数
        N, C, T, V, M = x.shape
        # 处理多人数据，取平均
        x = x.mean(dim=-1)  # [N, C, T, V]
        # 调整维度顺序以适应卷积层
        x = x.permute(0, 1, 3, 2)  # [N, C, V, T]
        # 如果提供了vars参数，使用这些参数而不是模型的参数
        # 这对MAML的内循环更新很重要
        if vars is not None:
            # 确保vars列表长度匹配模型参数数量
            if len(vars) != len(list(self.parameters())):
                print(f"警告: vars长度({len(vars)})与模型参数数量({len(list(self.parameters()))})不匹配!")
                # 如果不匹配，仍使用模型原有参数
                vars = None
        # 使用提供的vars或模型原有参数处理第一个卷积块
        if vars is None:
            x = F.relu(self.bn1(self.conv1(x)))  # [N, 64, V, T]
        else:
            idx = 0
            # 提取第一个卷积层参数
            w_conv1, b_conv1 = vars[idx], vars[idx+1]
            idx += 2
            # 提取第一个BN层参数
            w_bn1, b_bn1 = vars[idx], vars[idx+1]
            idx += 2
            
            # 使用提供的参数进行前向传播
            x = F.conv2d(x, w_conv1, b_conv1, padding=(0, 1))
            running_mean = self.bn1.running_mean
            running_var = self.bn1.running_var
            x = F.batch_norm(x, running_mean, running_var, w_bn1, b_bn1, training=self.training)
            x = F.relu(x)
        
        # 转换维度以用于图卷积
        x = x.permute(0, 2, 1, 3)  # [N, V, 64, T]
        
        # 图卷积操作
        x = torch.einsum('bij,bjct->bict', A, x)  # [N, V, 64, T]
        
        # 转回卷积格式
        x = x.permute(0, 2, 1, 3)  # [N, 64, V, T]
        
        # 使用提供的vars或模型原有参数处理第二个卷积块
        if vars is None:
            x = F.relu(self.bn2(self.conv2(x)))  # [N, out_channels, V, T]
        else:
            # 提取第二个卷积层参数
            w_conv2, b_conv2 = vars[idx], vars[idx+1]
            idx += 2
            # 提取第二个BN层参数
            w_bn2, b_bn2 = vars[idx], vars[idx+1]
            idx += 2
            
            # 使用提供的参数进行前向传播
            x = F.conv2d(x, w_conv2, b_conv2, padding=(0, 1))
            running_mean = self.bn2.running_mean
            running_var = self.bn2.running_var
            x = F.batch_norm(x, running_mean, running_var, w_bn2, b_bn2, training=self.training)
            x = F.relu(x)
        
        # 转换维度以用于第二次图卷积
        x = x.permute(0, 2, 1, 3)  # [N, V, out_channels, T]
        
        # 第二次图卷积操作
        x = torch.einsum('bij,bjct->bict', A, x)  # [N, V, out_channels, T]
        
        # 在时间维度上进行平均池化
        x = x.mean(dim=3)  # [N, V, out_channels]
        
        # 提取特征用于返回
        features = x  # [N, V, out_channels]
        
        # 展平并通过全连接层得到分类结果
        x_flat = x.reshape(N, -1)  # [N, V*out_channels]
        
        # 使用提供的vars或模型原有参数处理全连接层
        if vars is None:
            logits = self.fc(x_flat)  # [N, num_classes]
        else:
            # 提取全连接层参数
            w_fc, b_fc = vars[idx], vars[idx+1]
            # 使用提供的参数进行前向传播
            logits = F.linear(x_flat, w_fc, b_fc)

        return logits, features