from copy import deepcopy  # 保持这个
# from torch.utils.data import TensorDataset, DataLoader # 可能不需要，取决于数据加载方式
import numpy as np
import torch
from torch import nn
from torch import optim

import torch.nn.functional as F

# from maml.meta_first import lda_loss
# from maml.learner import Learner
# from maml.learner import Learner  # 确保这个导入是正确的
from maml.multi_head_mlp import net  # 确保这个导入是正确的
from maml.multi_head_mlp import GCN

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class Meta(nn.Module):
    """
    Meta Learner (适用于骨架数据和LDA损失)
    """

    def __init__(self, num_nodes=25, in_channels=3, update_lr=0.01, meta_lr=0.001, n_way=5, k_shot=5, k_query=15,
                 task_num=4, update_step=5):
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
            # todo 可学习的lda额外加入的
            num_person=2,  # m 应该是从数据维度 x_spt.size() 中获取的人数
            mlp_hidden_dim=128,  # 示例值
            mlp_output_dim=256,  # 示例值
            lda_output_dim=64,  # 示例值，可以根据 n_way-1 调整或固定
            n_way=self.n_way
        )
        self.GCN = GCN(
            n_way=self.n_way,
            in_channels=self.in_channels,
            out_channels = 256,  # 输出通道数可以根据需要调整
            num_nodes = self.num_nodes,
        )

        # 定义元优化器 (Meta Optimizer)
        # 将 Learner 和 GraphGenerator 的参数都加入优化
        self.meta_optim = torch.optim.Adam(
            list(self.net.parameters()) + list(self.GCN.parameters()),
            lr=self.meta_lr
        )

        self.eps = 1e-8  # 小的常数值，防止除零错误


    def lda(self, x, x_T, labels, N=None, M=None, T=None, topk=64):
        """
        x: [N*M*T, C] - 特征
    labels: [N*M*T] - 每帧标签
    return:
        loss: 标量 LDA 损失
        x_select: 选择后的重要帧特征
        """
        # 计算每个类别的均值向量
        total_frames, D = x_T.shape
        classes = torch.unique(labels)
        overall_mean = x_T.mean(dim=0, keepdim=True)  # [1, D]
        Sw = 0.0  # 类内散度
        Sb = 0.0  # 类间散度

        frame_scores = torch.zeros(x_T.shape[0], device=x_T.device)

        for cls in classes:
            mask = labels == cls
            cls_feats = x_T[mask]  # [n_c, D]
            cls_mean = cls_feats.mean(dim=0, keepdim=True)  # [1, D]

            # 类内散度：每个样本到本类均值的距离平方
            intra_dist = ((cls_feats - cls_mean) ** 2).sum(dim=1)  # [n_c]
            # 类间散度：类均值与总体均值的距离平方（同一个数）
            inter_dist = ((cls_mean - overall_mean) ** 2).sum().detach()

            # 加入总损失
            Sw += intra_dist.sum()
            Sb += cls_feats.shape[0] * inter_dist

            # 每帧的 importance score = inter / intra（越大越重要）
            frame_scores[mask] = inter_dist / (intra_dist + self.eps)
        
        # 重塑为 [N, M, T]
        frame_scores = frame_scores.view(N, M, T)  # 5，2，300
        # 对人物维度取平均得到视频帧分数
        VF_scores = frame_scores.mean(dim=1)   # [N, T] 即 [5, 300]
        topk_values, topk_idx = torch.topk(VF_scores, k=topk, dim=1)  #  # 获取每个样本的 topk 重要帧（5，64）
        values, topk_idx = topk_idx.sort(dim=1)# 升序排序


        # 首先reshape xt以便适应gather操作
        # 假设 x 形状为 [N, C, T, V, M]，我们需要将时间维度放到适合gather的位置
        x = x.permute(0, 3, 4, 1, 2)  # [N, V, M, C, T]
        
        # 扩展topk_idx为与x匹配的维度
        # 从 [N, topk] 变为 [N, V, M, C, topk]
        expanded_idx = topk_idx.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
            N, x.shape[1], x.shape[2], x.shape[3], topk)
        # 使用gather沿最后一个维度(时间维度)选择帧
        x = torch.gather(x, dim=-1, index=expanded_idx)
        x_select = x.permute(0, 3, 4, 1, 2)  # 回到 [N, C, topk, V, M]

        loss = Sw / (Sb + self.eps)

        return loss, x_select

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
            params = list(self.net.parameters()) + list(self.GCN.parameters())

            # 第一步：计算第i个任务在k=0时的损失和梯度
            s_T, s_adj = self.net(x_spt[i],vars=None,bn_training=True)  # 支持集前向传播
            spt_y = y_spt[i].view(setsz,1,1).expand(setsz, M, T).reshape(-1)
            lda_loss, sx_select = self.lda(x_spt[i], s_T, spt_y, setsz, M, T)

            logits = self.GCN(sx_select, s_adj, vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            loss = loss + lda_loss  # 将LDA损失加入总损失
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
                s_T, s_adj = self.net(x_spt[i],fast_weights, bn_training=True)  # 支持集前向传播
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
                    corrects[k + 1] = corrects[k + 1] + correct     # 累加正确预测数

        #计算所有任务的查询集平均损失
        loss_q  = losses_q[-1] / task_num
        # loss_q will be overwritten and just keep the loss_q on last update step.
        self.meta_optim.zero_grad() # 清空梯度
        loss_q.backward()
        self.meta_optim.step()  # 更新元参数
        accs = np.array(corrects) / (querysz * task_num)
        return accs


