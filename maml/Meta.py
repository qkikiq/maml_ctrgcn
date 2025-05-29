from copy import deepcopy  # 保持这个
# from torch.utils.data import TensorDataset, DataLoader # 可能不需要，取决于数据加载方式
import numpy as np
import torch
from torch import nn
from torch import optim

import torch.nn.functional as F
# from maml.learner import Learner
from maml.learner import Learner  # 确保这个导入是正确的
from maml.learner import GraphGenerator  # 确保这个导入是正确的


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

        # MAML 核心参数
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.task_num = task_num  # 元批次大小 (一个元批次中包含的任务数量)
        self.update_step = update_step  # 任务内学习的更新步数（训练时）


        self.net = Learner(
            num_nodes=self.num_nodes,
            in_channels=self.in_channels,
            out_channels=256,
            num_classes=self.n_way
        )

        # 创建图生成器实例，并将其参数加入元优化列表
        self.graph_generator = GraphGenerator(
            num_nodes = self.num_nodes, # 通常与Learner的节点数一致
            in_channels = self.in_channels # 通常与Learner的输入通道一致
        )

        # 定义元优化器 (Meta Optimizer)
        # 将 Learner 和 GraphGenerator 的参数都加入优化
        self.meta_optim = torch.optim.Adam(
            list(self.net.parameters()) + list(self.graph_generator.parameters()),
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
        losses_q = [0.0 for _ in range(self.update_step + 1)] # 使用浮点数初始化
        accs_q = [0.0 for _ in range(self.update_step + 1)]   # 使用浮点数初始化

        # 使用 self.graph_generator 生成邻接矩阵
        device = next(self.net.parameters()).device # 获取模型所在的设备
        adj_spt = self.graph_generator(x_spt.to(device))
        adj_qry = self.graph_generator(x_qry.to(device))

        # 存储每个任务在最后一次更新后，在查询集上提取的特征
        # Learner返回的features形状是 [N, V, out_channels]
        # self.net.fc.in_features // v 应该等于 out_channels (Learner中定义的)
        out_channels_dim = self.net.fc.in_features // v
        learned_features_qry_all_tasks = torch.zeros(task_num, querysz, v, out_channels_dim, device=device)

        # 遍历每个任务(每个episodic task)
        for i in range(task_num):
            # 提取当前任务的数据和邻接矩阵
            task_x_spt, task_y_spt = x_spt[i], y_spt[i]
            task_x_qry, task_y_qry = x_qry[i], y_qry[i]
            task_adj_spt, task_adj_qry = adj_spt[i], adj_qry[i]

            # 1. 初始评估 (step 0) - 使用原始元参数在查询集上评估
            with torch.no_grad():
                logits_q_orig, _ = self.net(task_x_qry, task_adj_qry, vars=None)
                loss_q_orig = self.loss_fn(logits_q_orig, task_y_qry)
                losses_q[0] += loss_q_orig.item() # 累加的是标量值

                pred_q_orig = F.softmax(logits_q_orig, dim=1).argmax(dim=1)
                correct_orig = torch.eq(pred_q_orig, task_y_qry).sum().item()
                accs_q[0] += correct_orig / querysz

            # 复制一份元参数作为内循环的起点
            fast_weights = [p.clone().detach() for p in self.net.parameters()]
            for p in fast_weights:
                p.requires_grad = True

            # 内循环更新
            for k in range(self.update_step):
                # a. 使用当前 fast_weights 在支持集上计算损失
                logits_spt, _ = self.net(task_x_spt, task_adj_spt, fast_weights)
                loss_spt = self.loss_fn(logits_spt, task_y_spt)

                # b. 计算梯度 (相对于 fast_weights)
                #    MAML允许在内循环中创建计算图，以便后续计算高阶导数，但这里我们只更新fast_weights本身
                #    如果 fast_weights 是从 self.net.parameters() clone() 并设置 requires_grad=True,
                #    那么 torch.autograd.grad(loss_spt, fast_weights) 是正确的。
                try:
                    grad = torch.autograd.grad(loss_spt, fast_weights, allow_unused=True)
                except RuntimeError as e:
                    print(f"警告: 任务 {i} 内循环步骤 {k} 计算梯度失败: {str(e)}. 跳过此更新。")
                    # 如果梯度计算失败，后续的fast_weights将不会更新，查询集评估将使用之前的权重
                    break # 中断当前任务的内循环

                # c. 更新 fast_weights (执行梯度下降)
                updated_fast_weights = []
                for p_idx, p_current in enumerate(fast_weights):
                    g = grad[p_idx] if grad is not None and p_idx < len(grad) else None
                    if g is None:
                        updated_fast_weights.append(p_current) # 保持原参数
                    else:
                        updated_fast_weights.append(p_current - self.update_lr * g)
                fast_weights = updated_fast_weights

                # d. 使用更新后的 fast_weights 在查询集上评估 (用于记录损失和准确率)
                #    这部分在 torch.no_grad() 下进行，因为它不直接参与元梯度的计算，
                #    元梯度是通过最后一步更新后的 fast_weights 在查询集上的损失计算的。
                with torch.no_grad():
                    logits_q, features_q_eval = self.net(task_x_qry, task_adj_qry, vars=fast_weights)
                    loss_q_step = self.loss_fn(logits_q, task_y_qry)
                    losses_q[k + 1] += loss_q_step.item() # k从0开始, 所以索引是1到update_step

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, task_y_qry).sum().item()
                    accs_q[k + 1] += correct / querysz
            
            # 内循环结束后
            # 使用最后得到的 fast_weights (经过 self.update_step 次更新) 在查询集上计算损失，这次需要计算图
            # 这是用于元优化的损失
            logits_q_final, features_q_final = self.net(task_x_qry, task_adj_qry, vars=fast_weights)
            loss_q_meta_grad = self.loss_fn(logits_q_final, task_y_qry)
            
            # 将用于元优化的损失累加到 losses_q 的最后一个元素 (如果内循环正常完成)
            # 如果内循环提前中断，这里的 loss_q_meta_grad 是基于中断前的 fast_weights
            # 为了简化，我们总是将这个最终的、带梯度的损失加到总的元损失上
            if i == 0: # 初始化元损失
                meta_loss_accumulator = loss_q_meta_grad
            else:
                meta_loss_accumulator = meta_loss_accumulator + loss_q_meta_grad

            learned_features_qry_all_tasks[i] = features_q_final.detach() # 保存最后一步的特征


        # 外循环: 使用累加的查询集损失进行元优化
        # losses_q 列表现在存储的是每个步骤的 *平均* 标量损失 (用于日志)
        # meta_loss_accumulator 存储的是所有任务最终查询集损失之和 (带梯度)
        
        if task_num > 0:
            final_meta_loss = meta_loss_accumulator / task_num
        else:
            final_meta_loss = torch.tensor(0.0, device=device, requires_grad=True) # 避免除以零

        # 执行元优化步骤
        self.meta_optim.zero_grad()
        try:
            final_meta_loss.backward()
            # 可选：梯度裁剪
            # torch.nn.utils.clip_grad_norm_(list(self.net.parameters()) + list(self.graph_generator.parameters()), max_norm=10)
            self.meta_optim.step()
        except RuntimeError as e:
            print(f"元优化步骤失败: {str(e)}")
            # 可以选择在这里记录错误，或者如果损失是nan/inf，则跳过优化步骤

        # 计算平均日志损失和准确率 (从标量累加值计算)
        avg_losses_q = [l / task_num if task_num > 0 else 0.0 for l in losses_q]
        avg_accs_q = [a / task_num if task_num > 0 else 0.0 for a in accs_q]

        # 检查最终的查询集损失 (用于日志的那个)
        if torch.isnan(final_meta_loss) or torch.isinf(final_meta_loss):
            print(f"警告: 最终元损失为 NaN/Inf: {final_meta_loss.item()}")
            # 实际用于反向传播的 final_meta_loss 如果是 nan/inf，梯度可能也是 nan/inf
            # 返回的损失值主要用于记录
            
        # 返回用于记录的最后一步的平均查询集损失和学习到的特征
        # meta_train.py 期望返回 (loss, learned_adj)
        # 我们返回 final_meta_loss.item() (标量) 和 learned_features_qry_all_tasks
        return final_meta_loss.item(), learned_features_qry_all_tasks,avg_accs_q

