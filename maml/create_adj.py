import torch
import torch.nn as nn

class Adj_Matrix(nn.Module):
    def __init__(self, mlp_input_dim, mlp_hidden_dim, num_nodes, use_sigmoid = False):
        """
        初始化邻接矩阵生成模块。

        参数:
        mlp_input_dim (int): MLP的输入特征维度。
                             例如，如果将 C, T, V, M 全部展平作为输入，则为 C*T*V*M。
        mlp_hidden_dim (int): MLP隐藏层的维度。
        V_nodes (int): 图中的节点数 (V)。
        use_sigmoid (bool, optional): 是否在MLP输出后使用Sigmoid激活函数。默认为 False。
        """
        super().__init__()

        self.V = num_nodes  # 存储节点数 V，用于重塑输出
        self.expected_mlp_input_dim = mlp_input_dim # 存储期望的MLP输入维度，用于校验

        # 定义 MLP 结构
        mlp_layers = [
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(), # 激活函数
            nn.Linear(mlp_hidden_dim, self.V * self.V) # 输出层，维度为 V*V
        ]

        if use_sigmoid:
            mlp_layers.append(nn.Sigmoid()) # 如果邻接值在0-1之间（如概率）

        self.mlp = nn.Sequential(*mlp_layers)

    def adj_matrix(self, data) :
        """
        从输入数据生成邻接矩阵。

        参数:
        data (torch.Tensor): 输入数据，形状为 [task_num, sample_num, C, T, V, M]。

        返回:
        torch.Tensor: 生成的邻接矩阵，形状为 [task_num, sample_num, V, V]。
        """
        task_num, sample_num, C, T, V, M = data.size()

        # 校验输入数据的V维度是否与模型初始化的V_nodes匹配
        if V != self.V:
            raise ValueError(
                f"输入数据的V维度 ({V}) 与模型初始化时指定的V_nodes ({self.V}) 不匹配。 "
                f"MLP的输出层根据初始化时的V_nodes固定为 V*V。"
            )
        mlp_ready_input = data.contiguous().view(task_num * sample_num, -1)
        # 校验展平后的特征维度是否与期望的mlp_input_dim匹配
        if mlp_ready_input.shape[1] != self.expected_mlp_input_dim:
            raise ValueError(
                f"展平后的输入特征维度 ({mlp_ready_input.shape[1]}) "
                f"与模型初始化时指定的 expected_mlp_input_dim ({self.expected_mlp_input_dim}) 不匹配。 "
                f"请确保 mlp_input_dim 正确对应您选择的特征提取方法（例如，C*T*V*M 完全展平）。"
            )
        # 2. 通过MLP进行处理
        # mlp_output 的形状: [task_num * sample_num, V * V]
        mlp_output = self.mlp(mlp_ready_input)

        # 3. 重塑MLP输出为邻接矩阵形式
        # 首先重塑为 [task_num * sample_num, V, V]
        # 然后重塑为 [task_num, sample_num, V, V]
        final_adj_matrix = mlp_output.view(task_num, sample_num, self.V, self.V)

        return final_adj_matrix

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        PyTorch nn.Module 的标准前向传播方法。
        """
        return self.adj_matrix(data)