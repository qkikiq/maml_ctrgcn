import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25   #节点数

# 定义自连接（self-link），即每个节点都与自身相连
self_link = [(i, i) for i in range(num_node)]

# 定义原始的入边索引（inward_ori_index）
# 每个元组 (i, j) 表示从 i 指向 j 的边
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

# 由于 Python 索引从 0 开始，而原始索引是从 1 开始的，所以将其转换为 0 开始的索引
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
# 定义出边（outward），即入边的反向边，每个 (i, j) 变成 (j, i)
outward = [(j, i) for (i, j) in inward]
# 定义邻接关系（neighbor），由入边和出边共同组成
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node   #节点数量
        self.self_link = self_link  #自连接
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        # 'spatial' 模式下，使用 tools 模块中的 get_spatial_graph 方法生成邻接矩阵
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
