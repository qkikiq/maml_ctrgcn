import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from maml.org_gcn import Model

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

        A = self.graph.A # 3,25,25

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
            graph_args=graph_args  # 传递图结构参数
        )  # 初始化GCN模型

        # 定义元优化器 (Meta Optimizer)
        self.meta_optim = optim.Adam(self.gcn.parameters(), lr=self.meta_lr)
        if not list(self.gcn.parameters()):  # 检查参数列表是否为空
            # 如果为空，使用以下方式获取参数
            self.meta_optim = optim.Adam(nn.Module.parameters(self.gcn), lr=self.meta_lr)


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
        :return: query set上的平均准确率
        """
        task_num, setsz, C, T, V, M = x_spt.size()
        querysz = x_qry.size(1)

        # 初始化列表存储每个更新步骤后的查询集损失和精度
        losses_q = [0.0 for _ in range(self.update_step + 1)]  # 使用浮点数初始化
        corrects = [0.0 for _ in range(self.update_step + 1)]  # 使用浮点数初始化

        # 遍历每个任务(每个episodic task)
        for i in range(task_num):
            logits, graph = self.gcn(x_spt[i], vars=None)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.gcn.parameters())
            fast_weights = []
            for g, p in zip(grad, self.gcn.parameters()):
                if g is None:
                    # 如果梯度为None，保持原参数不变
                    fast_weights.append(p)
                else:
                    # 否则正常更新参数
                    fast_weights.append(p - self.update_lr * g)

            #在第一次更新前评估模型
            with torch.no_grad():
                logits_q, graph_q = self.gcn(x_qry[i], self.gcn.parameters())
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # 在第一次更新后评估模型
            with torch.no_grad():
                logits_q, graph_q = self.gcn(x_qry[i], fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):  # update_step是内循环更新的次数
                logits, graph = self.gcn(x_spt[i], vars=fast_weights)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights,
                                           create_graph=True,
                                            retain_graph=True,
                                            allow_unused=True)
                
                new_fast_weights = []
                for g, p in zip(grad, fast_weights):
                    if g is None:
                        new_fast_weights.append(p)
                    else:
                        new_fast_weights.append(p - self.update_lr * g)
                fast_weights = new_fast_weights
                

                logits_q, graph_q = self.gcn(x_qry[i], vars=fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
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
        return  accs, graph_q


