from copy import deepcopy  # 保持这个
# from torch.utils.data import TensorDataset, DataLoader # 可能不需要，取决于数据加载方式
import numpy as np
import torch
from torch import nn
from torch import optim

import torch.nn.functional as F
from maml.gcn import SkeletonGraphGenerator  # 确保GraphGenerator在正确的路径下


class Meta(nn.Module):
    """
    Meta Learner (适用于骨架数据和LDA损失)
    """

    def __init__(self, num_nodes=25,num_person=2 , in_channels=3, hidden_dim=64,
                 update_lr=0.01, meta_lr=0.001, n_way=5, k_shot=5, k_query=15,
                 task_num=4, update_step=5):
        """
        构造函数
        :param args: 一个命名空间对象 (通常来自 argparse)，包含了所有从命令行和配置文件解析得到的参数。
                     其中 args.model_args 是一个字典，包含了模型特定参数 (来自YAML的 model_args 部分)。
                     args.n_way, args.k_spt, args.update_lr 等是MAML的核心参数，直接从args获取。
        """
        super(Meta, self).__init__()

        self.num_nodes = num_nodes
        # self.num_person = num_person
        # self.hidden_dim = hidden_dim  # 隐藏层维度
        self.in_channels = in_channels
        # MAML 核心参数
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.task_num = task_num  # 元批次大小 (一个元批次中包含的任务数量)
        self.update_step = update_step  # 任务内学习的更新步数（训练时）

        self.GraphGenerator = SkeletonGraphGenerator(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_dim=hidden_dim  ,
            num_person=num_person,
            n_way=n_way)  # 图生成器实例
        # 创建特征生成器实例，并将其参数加入元优化列表

    def lda_loss(self,x, labels, epsilon=1e-6):
        """
        features: [N, d]  批量样本特征
        labels: [N]       样本类别标签
        """
        device = x.device
        unique_labels = torch.unique(labels)
        N, d = x.shape

        overall_mean = x.mean(dim=0, keepdim=True)  # [1, d]

        Sw = torch.zeros((d, d), device=device)
        Sb = torch.zeros((d, d), device=device)

        for c in unique_labels:
            class_mask = (labels == c)
            Xc = x[class_mask]  # 类内样本
            Nc = Xc.shape[0]
            mean_c = Xc.mean(dim=0, keepdim=True)  # 类均值

            # 类内散度
            diff_w = Xc - mean_c  # [Nc, d]
            Sw += diff_w.T @ diff_w  # [d, d]

            # 类间散度
            mean_diff = (mean_c - overall_mean).T  # [d, 1]
            Sb += Nc * (mean_diff @ mean_diff.T)  # [d, d]

        loss = torch.trace(Sw) / (torch.trace(Sb) + epsilon)
        return loss

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
        task_num, setsz, C, T, V, M = x_spt.size()
        querysz = x_qry.size(1)

        # 初始化列表存储每个更新步骤后的查询集损失和精度
        losses_q = [0.0 for _ in range(self.update_step + 1)]  # 使用浮点数初始化
        corrects = [0.0 for _ in range(self.update_step + 1)]  # 使用浮点数初始化

        # 遍历每个任务(每个episodic task)
        for i in range(task_num):

            # 第一步：计算第i个任务在k=0时的损失和梯度
            outputs,y_hat, A = self.GraphGenerator(x_spt[i])  # 获取支持集的邻接矩阵  # output: [5,5]
            ce_loss = F.cross_entropy(y_hat, y_spt[i])
            loss_lda = self.lda_loss(outputs, y_spt[i])
            loss = ce_loss + 0.1 * loss_lda


            grad = torch.autograd.grad(loss, params, allow_unused=True)  # 计算梯度
            grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad, params)]
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, params)))

            with torch.no_grad():
                q_T, q_adj = self.net(x_qry[i], params, bn_training=True)
                qry_y = y_qry[i].view(querysz, 1, 1).expand(querysz, M, T).reshape(-1)
                _, qx_select = self.lda(x_qry[i], q_T, qry_y, querysz, M, T)
                logits_q = self.GCN(qx_select, q_adj)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                loss_q = loss_q + lda_loss
                losses_q[0] += loss_q  # 累加损失

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  # 获取预测结果
                correct = torch.eq(pred_q, y_qry[i]).sum().item()  # 计算正确预测数
                corrects[0] = corrects[0] + correct  # 累加正确预测数

            with torch.no_grad():
                q_T, q_adj = self.net(x_qry[i], fast_weights, bn_training=True)
                qry_y = y_qry[i].view(querysz, 1, 1).expand(querysz, M, T).reshape(-1)
                lda_loss, qx_select = self.lda(x_qry[i], q_T, qry_y, querysz, M, T)
                logits_q = self.GCN(qx_select, q_adj, fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                loss_q = loss_q + lda_loss
                losses_q[0] += loss_q  # 累加损失
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):  # update_step是内循环更新的次数
                s_T, s_adj = self.net(x_spt[i], fast_weights, bn_training=True)  # 支持集前向传播
                spt_y = y_spt[i].view(setsz, 1, 1).expand(setsz, M, T).reshape(-1)
                lda_loss, sx_select = self.lda(x_spt[i], s_T, spt_y, setsz, M, T)
                logits = self.GCN(sx_select, s_adj, vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                loss = loss + lda_loss  # 将LDA损失加入总损失
                grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)  # 计算梯度
                grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad, fast_weights)]
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                q_T, q_adj = self.net(x_qry[i], fast_weights, bn_training=True)
                qry_y = y_qry[i].view(querysz, 1, 1).expand(querysz, M, T).reshape(-1)
                lda_loss, qx_select = self.lda(x_qry[i], q_T, qry_y, querysz, M, T)
                logits_q = self.GCN(qx_select, q_adj, fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                loss_q = loss_q + lda_loss
                losses_q[k + 1] += loss_q  # 累加损失

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct  # 累加正确预测数

        # 计算所有任务的查询集平均损失
        loss_q = losses_q[-1] / task_num
        # loss_q will be overwritten and just keep the loss_q on last update step.
        self.meta_optim.zero_grad()  # 清空梯度
        loss_q.backward()
        self.meta_optim.step()  # 更新元参数
        accs = np.array(corrects) / (querysz * task_num)
        return accs


