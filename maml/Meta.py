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
            # 1. 初始评估
            logits, task_adj = self.net(x_spt[i], adj_spt[i], vars=None, bn_training=True)

            if logits.size(0) != setsz:  # 检查是否不匹配
                # 假设 logits 形状为 [N*M, C]，重塑为 [N, M, C]
                M = logits.size(0) // setsz
                logits = logits.view(setsz, M, -1)
                # 对 M 维度取平均，得到 [N, C]
                logits = logits.mean(dim=1)

            # 添加检查：确保 loss 是有效值
            loss = lda_loss(logits, y_spt[i])  # 计算支持集的LDA损失
            
            # 检查 loss 是否为 NaN 或 Inf，如果是则跳过此任务
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 任务 {i} 的损失为 NaN/Inf，跳过此任务")
                continue
                
            # 确保损失不为零向量，否则梯度计算会失败
            if loss.item() == 0:
                print(f"警告: 任务 {i} 的损失为零，跳过此任务")
                continue
                
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

        # 检查最终的查询集损失
        if torch.isnan(loss_q) or torch.isinf(loss_q):
            # 如果最终损失是 NaN，用一个小的常数替换它
            print("警告: 最终查询集损失为 NaN/Inf，使用小常数替代")
            loss_q = torch.tensor(0.01, device=x_spt.device, requires_grad=True)

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

    # 安全检查：如果类别太少或有非法值
    if num_classes < 2:  # LDA至少需要两个类
        return torch.tensor(0.01, device=device, requires_grad=True)
        
    # 检查嵌入向量是否包含非法值
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        print("警告: 嵌入向量包含 NaN 或 Inf 值")
        # 替换非法值
        embeddings = torch.where(torch.isnan(embeddings) | torch.isinf(embeddings), 
                                 torch.zeros_like(embeddings), 
                                 embeddings)

    # 计算类均值
    class_means = torch.zeros(num_classes, embedding_dim, device=device)
    class_counts = torch.zeros(num_classes, device=device)
    
    for i, label_val in enumerate(unique_labels):
        class_mask = (labels == label_val)
        class_counts[i] = class_mask.sum()
        
        # 安全检查：避免空类
        if class_counts[i] == 0:
            class_means[i] = torch.zeros(embedding_dim, device=device)
        else:
            class_means[i] = embeddings[class_mask].mean(dim=0)

    # 检查类均值是否有效
    if torch.isnan(class_means).any():
        print("警告: 类均值包含 NaN 值")
        return torch.tensor(0.01, device=device, requires_grad=True)

    # 1. 计算类内散度 (S_W) 和对角正则化
    s_w = torch.zeros(embedding_dim, embedding_dim, device=device)
    reg_strength = 1e-3  # 增加正则化强度
    
    for i, label_val in enumerate(unique_labels):
        class_mask = (labels == label_val)
        if class_mask.sum() > 0:  # 确保类不为空
            class_embeddings = embeddings[class_mask]
            mean_centered = class_embeddings - class_means[i].unsqueeze(0)
            s_w += mean_centered.t().mm(mean_centered)
    
    # 强化正则化以确保数值稳定性
    s_w += torch.eye(embedding_dim, device=device) * reg_strength
    
    # 2. 计算类间散度 (S_B)
    valid_samples = (class_counts > 0).sum()
    if valid_samples < 2:
        print("警告: 不足两个有效类别")
        return torch.tensor(0.01, device=device, requires_grad=True)
        
    overall_mean = embeddings.mean(dim=0)
    s_b = torch.zeros(embedding_dim, embedding_dim, device=device)
    
    for i, label_val in enumerate(unique_labels):
        if class_counts[i] > 0:  # 确保类不为空
            mean_diff = (class_means[i] - overall_mean).unsqueeze(1)
            s_b += class_counts[i] * mean_diff.mm(mean_diff.t())

    # 解决方程式，使用更稳定的方法
    try:
        # 使用特征值分解而不是直接求逆
        eigenvalues, eigenvectors = torch.linalg.eigh(s_w)
        # 处理可能的零或接近零的特征值
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        # 构建 S_W^(-1/2)
        s_w_inv_sqrt = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.t()
        # 计算 S_W^(-1/2) * S_B * S_W^(-1/2) 的特征值
        transformed_s_b = s_w_inv_sqrt @ s_b @ s_w_inv_sqrt
        # 使用特征值和来近似 trace(S_W^-1 * S_B)
        trace_s_w_inv_s_b = torch.linalg.eigvalsh(transformed_s_b).sum()
        loss = -trace_s_w_inv_s_b
    except Exception as e:
        print(f"LDA计算错误: {str(e)}")
        # 使用备用损失函数
        loss = torch.trace(s_w) - 0.1 * torch.trace(s_b)

    # 确保损失是有效值
    if torch.isnan(loss) or torch.isinf(loss):
        print("LDA损失是NaN或Inf，返回备用损失")
        return torch.tensor(0.01, device=device, requires_grad=True)

    return loss

