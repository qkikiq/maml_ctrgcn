import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from maml.org_gcn_1 import Model

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Meta(nn.Module):
    """
    Meta Learner (适用于骨架数据和LDA损失)
    """

    def __init__(self, num_person, num_point, update_lr, meta_lr, n_way,
                 task_num, update_step, graph=None, graph_args=dict(), in_channels=3):
        """
        构造函数
        """
        super(Meta, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)


        self.num_person = num_person
        self.num_point = num_point
        # MAML 核心参数
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.n_way = n_way
        self.task_num = task_num  # 元批次大小 (一个元批次中包含的任务数量)
        self.update_step = update_step  # 任务内学习的更新步数（训练时）
        self.in_channels = in_channels  # 输入通道数，通常为3（RGB图像）

        self.gcn = Model(
            in_channels=in_channels,
            num_person=self.num_person,
            num_point=num_point,  # 图节点数
            n_way=self.n_way,
            graph=graph,  # 传递图结构类的路径字符串
            graph_args=graph_args,  # 传递图结构参数
            residual = True
        )  # 初始化GCN模型

        # 定义元优化器 (Meta Optimizer)
        self.meta_optim = optim.Adam(self.gcn.parameters(), lr=self.meta_lr)


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

    def lda_loss(self, features, labels):
        """
        计算LDA损失，目标是最大化类间距离，最小化类内距离
        适用于在GPU上处理高维特征(1875维)

        Args:
            features: 特征向量 [batch_size, feature_dim]
            labels: 标签 [batch_size]

        Returns:
            lda_loss: LDA损失值
        """
        batch_size, feature_dim = features.size()
        device = features.device

        # 计算全局均值
        global_mean = torch.mean(features, dim=0)  # [feature_dim]

        # 初始化累加变量
        sw_trace = 0.0  # 类内散度矩阵的迹
        sb_trace = 0.0  # 类间散度矩阵的迹

        # 统计每个类的样本及其均值
        for c in range(self.n_way):
            # 找出当前类的所有样本
            indices = (labels == c).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue

            # 提取当前类的特征
            class_features = features[indices]

            # 计算类均值
            class_mean = torch.mean(class_features, dim=0)  # [feature_dim]

            # 计算类内散度的迹（不显式构建矩阵，直接计算迹）
            # tr(Sw_c) = tr((x_i - mu_c)(x_i - mu_c)^T)
            centered_features = class_features - class_mean.unsqueeze(0)  # [class_size, feature_dim]
            sw_trace += torch.sum(centered_features * centered_features)

            # 计算类间散度的迹（不显式构建矩阵）
            # tr(Sb_c) = n_c * tr((mu_c - mu)(mu_c - mu)^T)
            mean_diff = class_mean - global_mean  # [feature_dim]
            sb_trace += torch.sum(mean_diff * mean_diff)
        # 添加正则化项以确保数值稳定性
        sw_trace += 1e-4 * feature_dim
        # 损失函数：-tr(Sb)/tr(Sw)，加负号是因为我们要最小化损失（最大化类间/类内散度比）
        lda_loss = -sb_trace / (sw_trace + 1e-6)

        return lda_loss


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        task_num, setsz, C, T, V, M = x_spt.size()
        querysz = x_qry.size(1)
        # 初始化列表存储每个更新步骤后的查询集损失和精度
        losses_q = [0.0 for _ in range(self.update_step + 1)]  # 使用浮点数初始化
        corrects = [0.0 for _ in range(self.update_step + 1)]  # 使用浮点数初始化


        # LDA损失的权重系数
        lda_weight = 0.1  # 根据实验调整

        # 遍历每个任务(每个episodic task)
        for i in range(task_num):
            logits, graph = self.gcn(x_spt[i], vars=None, bn_training=True )
            ce_loss = F.cross_entropy(logits, y_spt[i])
            # 计算LDA损失
            lda = self.lda_loss(graph, y_spt[i])
            loss = ce_loss + lda_weight * lda
            grad = torch.autograd.grad(loss, self.gcn.parameters(), create_graph=True)  # 计算梯度
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.gcn.parameters())))

            #在第一次更新前评估模型
            with torch.no_grad():
                logits_q, graph_q = self.gcn(x_qry[i], self.gcn.parameters(),  bn_training=True)
                ce_loss = F.cross_entropy(logits_q, y_qry[i])
                lda = self.lda_loss(graph_q, y_qry[i])
                loss_q = ce_loss + lda_weight * lda

                losses_q[0] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0]  +=  correct

            # 在第一次更新后评估模型
            with torch.no_grad():
                logits_q, graph_q = self.gcn(x_qry[i], fast_weights , bn_training=True)
                ce_loss = F.cross_entropy(logits_q, y_qry[i])
                lda = self.lda_loss(graph_q, y_qry[i])
                loss_q = ce_loss + lda_weight * lda
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):  # update_step是内循环更新的次数
                logits, graph = self.gcn(x_spt[i], vars=fast_weights)
                ce_loss = F.cross_entropy(logits, y_spt[i])
                lda = self.lda_loss(graph, y_spt[i])
                loss = ce_loss + lda_weight * lda
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q, graph_q = self.gcn(x_qry[i], vars=fast_weights, bn_training=True)
                ce_loss = F.cross_entropy(logits_q, y_qry[i])
                lda = self.lda_loss(graph_q, y_spt[i])
                loss_q = ce_loss + lda_weight * lda
                losses_q[k + 1] += loss_q  # 累加损失

                with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()
                        corrects[k + 1] = corrects[k + 1] + correct  # 累加正确预测数

        # 计算所有任务的查询集平均损失
        loss_q = losses_q[-1] / task_num  #计算所有任务在最后一次内循环更新后的平均查询集损失
        # loss_q will be overwritten and just keep the loss_q on last update step.
        self.meta_optim.zero_grad()  # 清空梯度
        loss_q.backward()
        self.meta_optim.step()  # 更新元参数
        adj_matrix = self.gcn.vars[self.gcn.A_learn_idx].clone().detach()  # 提取学到的邻接矩阵参数

        accs = np.array(corrects) / (querysz * task_num)
        return  accs, adj_matrix


