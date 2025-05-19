from copy import deepcopy  # 保持这个
# from torch.utils.data import TensorDataset, DataLoader # 可能不需要，取决于数据加载方式
import numpy as np
import torch
from torch import nn
from torch import optim

from maml.learner import Learner


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
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
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
        self.net = Learner(config=learner_layer_config,
                           num_nodes=num_nodes_for_learner,
                           # num_subsets_A=num_subsets_A_for_learner
                           )

        # 定义元优化器 (Meta Optimizer)，用于更新 Learner 的参数 (self.net.parameters())
        # 例如，使用 Adam 优化器
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)

        # 你可能还需要一个损失函数，例如交叉熵损失，如果 Learner 的输出是 logits
        # 或者像 LDA 损失，如果 Learner 的输出是嵌入向量
        # self.loss_fn = nn.CrossEntropyLoss() # 或者其他适合你任务的损失函数

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

    def forward(self, x_spt, y_spt, adj_spt, x_qry, y_qry, adj_qry):  # 增加了adj_spt, adj_qry参数
        """
        元训练步骤
        :param x_spt:   [b, setsz_spt, feature_dim] 或 [b, setsz_spt, num_nodes, node_feature_dim] (取决于Learner输入)
        :param y_spt:   [b, setsz_spt] (标签)
        :param adj_spt: [b, setsz_spt, num_nodes, num_nodes] (支持集邻接矩阵)
        :param x_qry:   [b, setsz_qry, feature_dim] 或 [b, setsz_qry, num_nodes, node_feature_dim]
        :param y_qry:   [b, setsz_qry] (标签)
        :param adj_qry: [b, setsz_qry, num_nodes, num_nodes] (查询集邻接矩阵)
        :return: query set上的平均准确率 (在每个内部更新步骤之后)
        """
        task_num, setsz, c, t,v,m = x_spt.size()
        task_num,querysz, c, t,v,m = x_qry.size()
        # 获取任务数量(meta-batch size)，即有多少个独立的任务需要元学习
        task_num = x_spt.size(0)

        # 获取每个任务的查询集大小(样本数)
        querysz = x_qry.size(1)

        # 初始化列表存储每个更新步骤后的查询集损失
        losses_q = [0 for _ in range(self.update_step + 1)]

        # 初始化存储学习后邻接矩阵的变量
        learned_adj = adj_qry.clone()  # 初始化为查询集的邻接矩阵


        # 遍历每个任务(每个episodic task)
        for i in range(task_num):
            # 1. 初始评估  - 在任何参数更新前计算模型性能基准
            #    注意: 这里的self.net.parameters()是元参数theta
            #    BN层的training状态在MAML中通常保持True，因为它也参与内循环的适应，或者，如果BN层的统计数据不希望在内循环中改变，可以考虑更复杂的BN处理
            # 用当前参数获取支持集的嵌入表示
            logits, task_adj = self.net(x_spt[i], adj_spt[i], vars=None, bn_training=True)

            if logits.size(0) != setsz:  # 检查是否不匹配
                # 假设 logits 形状为 [N*M, C]，重塑为 [N, M, C]
                M = logits.size(0) // setsz
                logits = logits.view(setsz, M, -1)
                # 对 M 维度取平均，得到 [N, C]
                logits = logits.mean(dim=1)

            loss = lda_loss(logits, y_spt[i])  # 计算支持集的LDA损失
            grad = torch.autograd.grad(loss, self.net.parameters())  # 计算梯度
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))  # 使用梯度和学习率更新权重
            # 在第一次更新前评估模型
            with torch.no_grad():  # 不需要梯度计算，节省内存
                # 获取查询集的初始嵌入表示(用原始元参数)
                #todo 维度更改
                logits_q,task_adj2 = self.net(x_qry[i], adj_qry[i], vars=self.net.parameters(), bn_training=True)

                if logits_q.size(0) != querysz:  # 检查是否不匹配
                    # 假设 logits 形状为 [N*M, C]，重塑为 [N, M, C]
                    M = logits_q.size(0) // querysz
                    logits_q = logits_q.view(querysz, M, -1)
                    # 对 M 维度取平均，得到 [N, C]
                    logits_q = logits_q.mean(dim=1)

                loss_q = lda_loss(logits_q, y_qry[i])
                losses_q[0] += loss_q  # 累加损失

            # 2b. 使用更新后的参数(fast_weights)在查询集上评估第一次更新后的效果
            with torch.no_grad():
                # 使用fast_weights获取查询集的嵌入表示
                logits_q,task_adj2 = self.net(x_qry[i], adj_qry[i], vars=fast_weights, bn_training=True)

                if logits_q.size(0) != querysz:  # 检查是否不匹配
                    # 假设 logits_q 形状为 [N*M, C]，重塑为 [N, M, C]
                    M = logits_q.size(0) // querysz
                    logits_q = logits_q.view(querysz, M, -1)
                    # 对 M 维度取平均，得到 [N, C]
                    logits_q = logits_q.mean(dim=1)

                # 计算查询集上的LDA损失
                loss_q = lda_loss(logits_q, y_qry[i])
                # 累加第1步更新后的查询集损失
                losses_q[1] += loss_q  # 累加第i个任务在第1步更新后的查询集损失

            # 2c. 进行更多内循环更新步骤(第2步到第update_step步)
            for k in range(1, self.update_step):
                # 使用当前fast_weights计算支持集嵌入
                logits,task_adj = self.net(x_spt[i], adj_spt[i], vars=fast_weights, bn_training=True)

                if logits.size(0) != setsz:  # 检查是否不匹配
                    # 假设 logits 形状为 [N*M, C]，重塑为 [N, M, C]
                    M = logits.size(0) // setsz
                    logits = logits.view(setsz, M, -1)
                    # 对 M 维度取平均，得到 [N, C]
                    logits = logits.mean(dim=1)

                # 计算当前支持集的LDA损失
                loss = lda_loss(logits, y_spt[i])

                # 计算损失对fast_weights的梯度(注意这里是对fast_weights求导)
                # grad = torch.autograd.grad(loss, fast_weights)  # 注意这里是对 fast_weights 求导
                grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
                # 更新fast_weights：θ'k+1 = θ'k - α∇θ'kL
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                # 使用更新后的fast_weights在查询集上评估
                logits_q, task_adj2 = self.net(x_qry[i], adj_qry[i], vars=fast_weights, bn_training=True)

                if logits_q.size(0) != querysz:  # 检查是否不匹配
                    # 假设 logits_q 形状为 [N*M, C]，重塑为 [N, M, C]
                    M = logits_q.size(0) // querysz
                    logits_q = logits_q.view(querysz, M, -1)
                    # 对 M 维度取平均，得到 [N, C]
                    logits_q = logits_q.mean(dim=1)

                # 计算查询集上的LDA损失
                loss_q = lda_loss(logits_q, y_qry[i])

                # 累加当前步骤的查询集损失
                losses_q[k + 1] += loss_q  # 累加第i个任务在第 k+1 步更新后的查询集损失
        if self.update_step > 0:
            # 最后一次更新后使用 fast_weights 在查询集上运行模型
            _, final_adj = self.net(x_qry[i], adj_qry[i], vars=fast_weights, bn_training=True)
            learned_adj[i] = final_adj


        #    使用所有任务在最后一步内循环更新后 (self.update_step) 的查询集损失进行元优化
        loss_q = losses_q[self.update_step] / task_num  # 平均查询集损失

        self.meta_optim.zero_grad()
        # 反向传播计算梯度
        loss_q.backward()  # PyTorch会自动处理链式法则，将梯度反向传播到原始的self.net.parameters()
        # self.clip_grad_by_norm_(self.net.parameters(), 10) # 可选的梯度裁剪
        # 使用元优化器更新元参数
        self.meta_optim.step()

        return loss_q,learned_adj

    # def finetunning(self, x_spt, y_spt, adj_spt, x_qry, y_qry, adj_qry):
    #     """
    #     在新任务上进行微调和评估 (测试阶段)
    #     :param x_spt:   [b, setsz_spt, C, T, V, M] - 支持集特征
    #     :param y_spt:   [b, setsz_spt] - 支持集标签
    #     :param adj_spt: [b, setsz_spt, num_subsets_A, num_nodes, num_nodes] - 支持集邻接矩阵
    #     :param x_qry:   [b, setsz_qry, C, T, V, M] - 查询集特征
    #     :param y_qry:   [b, setsz_qry] - 查询集标签
    #     :param adj_qry: [b, setsz_qry, num_subsets_A, num_nodes, num_nodes] - 查询集邻接矩阵
    #     :return: 准确率列表和学习后的邻接矩阵
    #     """
    #     task_num, setsz, c, t, v, m = x_spt.size()
    #     task_num, querysz, c, t, v, m = x_qry.size()
    #
    #     # 初始化准确率列表和损失列表
    #     losses_q = [0 for _ in range(self.update_step_test + 1)]
    #     corrects = [0 for _ in range(self.update_step_test + 1)]
    #
    #     # 初始化存储学习后邻接矩阵的变量
    #     learned_adj = adj_qry.clone()
    #
    #     # 深拷贝网络，避免修改原始元参数
    #     net = deepcopy(self.net)
    #
    #     # 遍历每个任务
    #     for i in range(task_num):
    #         # 1. 初始评估 - 在任何参数更新前的基准性能
    #         logits, task_adj = net(x_spt[i], adj_spt[i], vars=None, bn_training=True)
    #
    #         if logits.size(0) != setsz:
    #             M = logits.size(0) // setsz
    #             logits = logits.view(setsz, M, -1)
    #             logits = logits.mean(dim=1)
    #
    #         loss = lda_loss(logits, y_spt[i])
    #         grad = torch.autograd.grad(loss, net.parameters())
    #         fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
    #
    #         # 在初始参数上评估查询集
    #         with torch.no_grad():
    #             logits_q, _ = net(x_qry[i], adj_qry[i], vars=net.parameters(), bn_training=True)
    #
    #             if logits_q.size(0) != querysz:
    #                 M = logits_q.size(0) // querysz
    #                 logits_q = logits_q.view(querysz, M, -1)
    #                 logits_q = logits_q.mean(dim=1)
    #
    #             loss_q = lda_loss(logits_q, y_qry[i])
    #             losses_q[0] += loss_q
    #
    #             # 可以添加准确率计算，这里使用零占位符
    #             corrects[0] += 0
    #
    #         # 2. 使用更新后的参数评估第一次更新
    #         with torch.no_grad():
    #             logits_q, _ = net(x_qry[i], adj_qry[i], vars=fast_weights, bn_training=True)
    #
    #             if logits_q.size(0) != querysz:
    #                 M = logits_q.size(0) // querysz
    #                 logits_q = logits_q.view(querysz, M, -1)
    #                 logits_q = logits_q.mean(dim=1)
    #
    #             loss_q = lda_loss(logits_q, y_qry[i])
    #             losses_q[1] += loss_q
    #
    #             # 可以添加准确率计算，这里使用零占位符
    #             corrects[1] += 0
    #
    #         # 3. 后续更新步骤
    #         for k in range(1, self.update_step_test):
    #             # 使用当前fast_weights计算支持集嵌入
    #             logits, _ = net(x_spt[i], adj_spt[i], vars=fast_weights, bn_training=True)
    #
    #             if logits.size(0) != setsz:
    #                 M = logits.size(0) // setsz
    #                 logits = logits.view(setsz, M, -1)
    #                 logits = logits.mean(dim=1)
    #
    #             loss = lda_loss(logits, y_spt[i])
    #             grad = torch.autograd.grad(loss, fast_weights)
    #             fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
    #
    #             # 评估当前步骤
    #             with torch.no_grad():
    #                 logits_q, _ = net(x_qry[i], adj_qry[i], vars=fast_weights, bn_training=True)
    #
    #                 if logits_q.size(0) != querysz:
    #                     M = logits_q.size(0) // querysz
    #                     logits_q = logits_q.view(querysz, M, -1)
    #                     logits_q = logits_q.mean(dim=1)
    #
    #                 loss_q = lda_loss(logits_q, y_qry[i])
    #                 losses_q[k + 1] += loss_q
    #
    #                 # 可以添加准确率计算，这里使用零占位符
    #                 corrects[k + 1] += 0
    #
    #         # 最后一次更新后获取学习到的邻接矩阵
    #         if self.update_step_test > 0:
    #             _, final_adj = net(x_qry[i], adj_qry[i], vars=fast_weights, bn_training=True)
    #             learned_adj[i] = final_adj
    #
    #     # 计算平均损失和准确率
    #     losses_q = [loss / task_num for loss in losses_q]
    #     accs = [correct / (task_num * querysz) if task_num * querysz > 0 else 0 for correct in corrects]
    #
    #     return losses_q, accs, learned_adj



def lda_loss(embeddings, labels):
    # embeddings: [batch_size, embedding_dim]
    # labels: [batch_size]
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    embedding_dim = embeddings.size(1)
    device = embeddings.device

    if num_classes < 2: # LDA至少需要两个类
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 计算类均值
    class_means = torch.zeros(num_classes, embedding_dim, device=device)
    for i, label_val in enumerate(unique_labels):
        class_means[i] = embeddings[labels == label_val].mean(dim=0)

    # 1. 计算类内散度 (S_W)
    s_w = torch.zeros(embedding_dim, embedding_dim, device=device)
    for i, label_val in enumerate(unique_labels):
        class_embeddings = embeddings[labels == label_val]
        mean_centered_embeddings = class_embeddings - class_means[i].unsqueeze(0)
        s_w += mean_centered_embeddings.t().mm(mean_centered_embeddings)

    # 2. 计算类间散度 (S_B)
    overall_mean = embeddings.mean(dim=0)
    s_b = torch.zeros(embedding_dim, embedding_dim, device=device)
    for i, label_val in enumerate(unique_labels):
        num_samples_class = (labels == label_val).sum()
        mean_diff = (class_means[i] - overall_mean).unsqueeze(1) # [embedding_dim, 1]
        s_b += num_samples_class * mean_diff.mm(mean_diff.t())

    # 为了数值稳定性，可以给S_W的对角线加上一个小的epsilon
    s_w += torch.eye(embedding_dim, device=device) * 1e-4

    try:
        # 计算 S_W_inv * S_B
        s_w_inv_s_b = torch.linalg.solve(s_w, s_b) # 更稳定
        # s_w_inv_s_b = torch.linalg.inv(s_w).mm(s_b)
        loss = -torch.trace(s_w_inv_s_b)
    except torch.linalg.LinAlgError: # 如果S_W奇异
        # print("S_W is singular, using alternative LDA loss")
        loss = torch.trace(s_w) - torch.trace(s_b) # 备用损失

    if torch.isnan(loss) or torch.isinf(loss):
        # print("LDA loss is NaN or Inf, returning zero loss for this batch.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss

