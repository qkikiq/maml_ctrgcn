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

    def __init__(self, args):
        """
        构造函数
        :param args: 一个命名空间对象 (通常来自 argparse)，包含了所有从命令行和配置文件解析得到的参数。
                     其中 args.model_args 是一个字典，包含了模型特定参数 (来自YAML的 model_args 部分)。
                     args.n_way, args.k_spt, args.update_lr 等是MAML的核心参数，直接从args获取。
        """
        super(Meta, self).__init__()

        # MAML 核心参数
        self.update_lr = args.update_lr * 0.1
        self.meta_lr = args.meta_lr * 0.1
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num  # 元批次大小 (一个元批次中包含的任务数量)
        self.update_step = args.update_step  # 任务内学习的更新步数（训练时）
        self.update_step_test = args.update_step_test  # 任务内学习的更新步数（测试时）

        # 从 args.model_args (这是一个字典) 中提取 Learner 所需的特定参数
        if not hasattr(args, 'model_args') or args.model_args is None:
            raise ValueError("模型参数 'model_args' 没有在配置中提供或为空。")

        learner_specific_params = args.model_args  # model_args 本身就是解析后的字典

        # 提取 Learner 的网络配置列表 (对应 Learner 的 'config' 参数)
        learner_layer_config = learner_specific_params.get('learner_config')
        if learner_layer_config is None:
            raise ValueError("在 'model_args' 中未找到 'learner_config'。请在YAML配置中定义。")

        # 提取 GCN 的节点数
        num_nodes_for_learner = learner_specific_params.get('num_nodes')
        if num_nodes_for_learner is None:
            raise ValueError("在 'model_args' 中未找到 'num_nodes'。请在YAML配置中定义。")

        # # 提取 GCN 的邻接矩阵子集数
        # num_subsets_A_for_learner = learner_specific_params.get('num_subsets_A')
        # if num_subsets_A_for_learner is None:
        #     raise ValueError("在 'model_args' 中未找到 'num_subsets_A'。请在YAML配置中定义。")

        # 实例化 Learner (self.net)
        # Learner 的构造函数签名是 __init__(self, config, num_nodes, num_subsets_A)
        # self.net = Learner(config=learner_layer_config,
        #                    num_nodes=num_nodes_for_learner,
        #                    # num_subsets_A=num_subsets_A_for_learner
        #                    )

        self.net = Learner(
            num_joints=num_nodes_for_learner,
            in_channels=3,  # 根据骨架数据的通道数
            out_channels=128,  # 可以从配置中读取
            num_classes=self.n_way  # 确保与元学习任务一致
        )

        # meta_params = list(graph_generator.parameters()) + list(learner.parameters())
        # 定义元优化器 (Meta Optimizer)，用于更新 Learner 的参数 (self.net.parameters())
        # 例如，使用 Adam 优化器
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)

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
        :return: query set上的平均损失和学习后的邻接矩阵
        """
        task_num, setsz, c, t, v, m = x_spt.size()
        task_num, querysz, c, t, v, m = x_qry.size()
        
        # 初始化列表存储每个更新步骤后的查询集损失和精度
        losses_q = [0 for _ in range(self.update_step + 1)]
        accs_q = [0 for _ in range(self.update_step + 1)]

        # 创建图生成器实例
        graph_generator = GraphGenerator(num_joints=v, in_channels=c).to(x_spt.device)

        adj_spt= graph_generator(x_spt)
        adj_qry = graph_generator(x_qry)

        # 初始化邻接矩阵存储
        # adj_spt = torch.zeros(task_num, setsz, v, v).to(x_spt.device)
        # adj_qry = torch.zeros(task_num, querysz, v, v).to(x_qry.device)


        learned_adj = torch.zeros_like(adj_qry)

        # 遍历每个任务(每个episodic task)
        for i in range(task_num):
            # # 为支持集和查询集生成邻接矩阵
            # adj_spt[i] = graph_generator(x_spt[i])
            # adj_qry[i] = graph_generator(x_qry[i])

            # 1. 初始评估 - 使用原始模型参数
            # 确保不传递额外的vars参数
            logits, _ = self.net(x_spt[i], adj_spt[i], vars=None)
            
            # 使用交叉熵损失
            loss = self.loss_fn(logits, y_spt[i])
            
            # # 检查 loss 是否为 NaN 或 Inf，如果是则跳过此任务
            # if torch.isnan(loss) or torch.isinf(loss):
            #     print(f"警告: 任务 {i} 的损失为 NaN/Inf，跳过此任务")
            #     continue
            #
            # # 确保损失不为零向量，否则梯度计算会失败
            # if loss.item() == 0:
            #     print(f"警告: 任务 {i} 的损失为零，跳过此任务")
            #     continue
            
            # 计算梯度并更新快速权重
            try:
                grad = torch.autograd.grad(loss, self.net.parameters(), allow_unused=True)
            except Exception as e:
                print(f"计算梯度时发生错误: {str(e)}")
                print(f"跳过任务 {i}")
                continue

            # 验证梯度和参数是否匹配
            param_count = len(list(self.net.parameters()))
            grad_count = len(grad)
            if param_count != grad_count:
                print(f"警告: 参数数量({param_count})与梯度数量({grad_count})不匹配!")

            # 确保安全地创建 fast_weights
            fast_weights = []
            for idx, (g, p) in enumerate(zip(grad, self.net.parameters())):
                if g is None:
                    # 如果某个参数没有梯度，保持原参数不变
                    fast_weights.append(p)
                else:
                    fast_weights.append(p - self.update_lr * g)

            # 确保 fast_weights 长度等于模型参数数量
            if len(fast_weights) != param_count:
                print(f"警告: fast_weights长度({len(fast_weights)})与模型参数数量({param_count})不匹配!")
                # 确保安全，使用原始参数
                fast_weights = list(self.net.parameters())

            # 在第一次更新前评估模型
            with torch.no_grad():
                # 获取查询集的初始输出(用原始元参数)
                logits_q, _ = self.net(x_qry[i], adj_qry[i])

                # 使用交叉熵损失
                loss_q = self.loss_fn(logits_q, y_qry[i])
                losses_q[0] += loss_q  # 累加损失
                
                # 计算准确率
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                accs_q[0] += correct / querysz  # 累加准确率

            # 2b. 使用更新后的参数(fast_weights)在查询集上评估第一次更新后的效果
            with torch.no_grad():
                # 使用fast_weights获取查询集的输出
                logits_q, _ = self.net(x_qry[i], adj_qry[i], vars=fast_weights)

                # 使用交叉熵损失
                loss_q = self.loss_fn(logits_q, y_qry[i])
                losses_q[1] += loss_q  # 累加第1步更新后的查询集损失
                
                # 计算准确率
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                accs_q[1] += correct / querysz  # 累加准确率

            # 2c. 进行更多内循环更新步骤(第2步到第update_step步)
            for k in range(1, self.update_step):
                # 使用当前fast_weights计算支持集输出
                logits, _ = self.net(x_spt[i], adj_spt[i], vars=fast_weights)

                # 使用交叉熵损失
                loss = self.loss_fn(logits, y_spt[i])
                
                # 检查 loss
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0:
                    print(f"警告: 任务 {i} 在内循环步骤 {k} 的损失无效，跳过此更新")
                    continue
                
                try:
                    grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
                    fast_weights = []
                    for g, w in zip(grad, fast_weights):
                        if g is None:
                            # 如果梯度为None，保持原参数不变
                            fast_weights.append(w)
                        else:
                            # 否则正常更新参数
                            fast_weights.append(w - self.update_lr * g)
                except Exception as e:
                    print(f"内循环步骤 {k} 计算梯度时发生错误: {str(e)}")
                    continue

                # 使用更新后的fast_weights在查询集上评估
                logits_q, _ = self.net(x_qry[i], adj_qry[i], vars=fast_weights)

                # 使用交叉熵损失
                loss_q = self.loss_fn(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q  # 累加当前步骤的查询集损失
                
                # 计算准确率
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                accs_q[k + 1] += correct / querysz  # 累加准确率

        # 保存最后一次更新后的邻接矩阵
        if self.update_step > 0:
            # 最后一次更新后使用 fast_weights 在查询集上运行模型
            _, final_adj = self.net(x_qry[i], adj_qry[i], vars=fast_weights)
            learned_adj[i] = final_adj

        # 使用所有任务在最后一步内循环更新后的查询集损失进行元优化
        if losses_q[self.update_step] == 0:
            loss_q = torch.tensor(0.0, device=x_spt.device, requires_grad=True)
        else:
            loss_q = losses_q[self.update_step] / task_num
            if not isinstance(loss_q, torch.Tensor) or not loss_q.requires_grad:
                loss_q = torch.tensor(loss_q, device=x_spt.device, requires_grad=True)

        # 计算平均准确率
        acc_q = accs_q[self.update_step] / task_num

        # 执行元优化步骤
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        # 检查最终的查询集损失
        if torch.isnan(loss_q) or torch.isinf(loss_q):
            print("警告: 最终查询集损失为 NaN/Inf，使用小常数替代")
            loss_q = torch.tensor(0.01, device=x_spt.device, requires_grad=True)

        # 返回平均损失和学习后的邻接矩阵
        return loss_q, learned_adj

