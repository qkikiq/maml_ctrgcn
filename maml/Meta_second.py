from copy import deepcopy  # 保持这个
# from torch.utils.data import TensorDataset, DataLoader # 可能不需要，取决于数据加载方式
import numpy as np
import torch
from torch import nn
from torch import optim

import torch.nn.functional as F
# from maml.learner import Learner
# from maml.learner import Learner  # 确保这个导入是正确的
from maml.Graph_Generator import net  # 确保这个导入是正确的

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class Meta(nn.Module):
    """
    Meta Learner (适用于骨架数据和LDA损失)
    """

    def __init__(self, num_nodes=25,in_channels = 3, update_lr=0.01, meta_lr=0.001, n_way=5, k_shot=5, k_query=15, task_num=4,update_step=5):
        """
        构造函数
        :param args: 一个命名空间对象 (通常来自 argparse)，包含了所有从命令行和配置文件解析得到的参数。
                     其中 args.model_args 是一个字典，包含了模型特定参数 (来自YAML的 model_args 部分)。
                     args.n_way, args.k_spt, args.update_lr 等是MAML的核心参数，直接从args获取。
        """
        super(Meta, self).__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        # MAML 核心参数
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.task_num = task_num  # 元批次大小 (一个元批次中包含的任务数量)
        self.update_step = update_step  # 任务内学习的更新步数（训练时）

        # 创建特征生成器实例，并将其参数加入元优化列表
        self.net = net(
            num_nodes=self.num_nodes,  # 通常与Learner的节点数一致
            in_channels=self.in_channels,  # 通常与Learner的输入通道一致
            #todo 可学习的lda额外加入的
            num_person=2, # m 应该是从数据维度 x_spt.size() 中获取的人数
            mlp_hidden_dim=128, # 示例值
            mlp_output_dim=256, # 示例值
            lda_output_dim=64,  # 示例值，可以根据 n_way-1 调整或固定
            n_way=self.n_way
        )

        # 定义元优化器 (Meta Optimizer)
        # 将 Learner 和 GraphGenerator 的参数都加入优化
        self.meta_optim = torch.optim.Adam(
            list(self.net.parameters()),
            lr=self.meta_lr
        )

        # 添加交叉熵损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def clip_grad_by_norm_(self, grad, max_norm):  # 这个函数可以保持不变
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            if g is None:  # 梯度可能为None，如果某些参数没有参与计算
                continue
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        if counter == 0:
            return 0
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                if g is None:
                    continue
                g.data.mul_(clip_coef)
        return total_norm / counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        元训练步骤
        :param x_spt:   [b, setsz_spt, c, t, v, m] 支持集输入
        :param y_spt:   [b, setsz_spt] 支持集标签
        :param x_qry:   [b, setsz_qry, c, t, v, m] 查询集输入
        :param y_qry:   [b, setsz_qry] 查询集标签
        :return: query set上的平均损失和学习后的特征 (features)
        """
        task_num, setsz, c, t, v, m = x_spt.size()
        querysz = x_qry.size(1)

        # 初始化列表存储每个更新步骤后的查询集损失和精度
        losses_q = [0.0 for _ in range(self.update_step + 1)]  # 使用浮点数初始化
        corrects = [0.0 for _ in range(self.update_step + 1)]  # 使用浮点数初始化

        # 遍历每个任务(每个episodic task)
        for i in range(task_num):

            
            # 第一步：计算第i个任务在k=0时的损失和梯度
            logits = self.net(x_spt[i],y_spt[i])  # 支持集前向传播



            loss = F.cross_entropy(logits, y_spt[i])  # 计算交叉熵损失
            grad = torch.autograd.grad(loss, self.net.parameters())  # 计算梯度
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))  # 使用梯度和学习率更新权重

            # 1. 初始评估 (step 0) - 使用原始元参数在查询集上评估
            with torch.no_grad():
                logits_q, _ = self.net(x_qry[i], vars=None)
                loss_q = self.loss_fn(logits_q, y_qry[i])
                losses_q[0] += loss_q.item()  # 累加的是标量值

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct   # 累加正确预测数

            # 在第一次更新后评估模型
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            # 内循环更新
            for k in range(self.update_step):
                # a. 使用当前 fast_weights 在支持集上计算损失
                logits, _ = self.net(x_spt[i],fast_weights)
                loss = self.loss_fn(logits, y_spt[i])
                # b. 计算梯度 (相对于 fast_weights)
                grad = torch.autograd.grad(loss, fast_weights)
                # c. 更新 fast_weights (执行梯度下降)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                # d. 使用更新后的 fast_weights 在查询集上评估 (用于记录损失和准确率)

                logits_q, _ = self.net(x_qry[i], fast_weights)
                loss_q = self.loss_fn(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q   #查询集损失

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        # 内循环结束后 (或因错误中断后)
        # 使用最后得到的 fast_weights (经过 self.update_step 次更新) 在查询集上计算损失，这次需要计算图
        # 这是用于元优化的损失
        loss_q = losses_q[-1] / task_num
        # 优化元参数
        # optimize theta parameters
        self.meta_optim.zero_grad()  # 清空梯度
        loss_q.backward()
        self.meta_optim.step()  # 更新元参数
        accs = np.array(corrects) / (querysz * task_num)

        return accs

