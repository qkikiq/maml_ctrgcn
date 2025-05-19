import numpy as np

def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)  # 计算每列的度（入度）
    h, w = A.shape  # 获取矩阵形状
    Dn = np.zeros((w, w))  # 创建一个 w x w 的零矩阵
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)  # 计算 D^-1
    AD = np.dot(A, Dn)  # 计算 A * D^-1
    return AD  # 返回归一化后的邻接矩阵

#构造图
def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)   #自连接
    In = normalize_digraph(edge2mat(inward, num_node))   #归一化入边邻接矩阵
    Out = normalize_digraph(edge2mat(outward, num_node))   #归一化出边邻接矩阵
    A = np.stack((I, In, Out))  # 将三个邻接矩阵合并成 3 维矩阵
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)  # 确保 A 是 NumPy 数组
    I = np.eye(len(A), dtype=A.dtype)  # 创建单位矩阵
    if k == 0:
        return I  # 0 阶邻接矩阵为单位矩阵
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)  # 计算 k 阶邻接矩阵
    if with_self:
        Ak += (self_factor * I)  # 如果需要自连接，加入 self_factor
    return Ak  # 返回 k 阶邻接矩阵


def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)  # 自连接邻接矩阵
    A1 = edge2mat(inward, num_node)  # 一阶入边邻接矩阵
    A2 = edge2mat(outward, num_node)  # 一阶出边邻接矩阵
    A3 = k_adjacency(A1, 2)  # 二阶入边邻接矩阵
    A4 = k_adjacency(A2, 2)  # 二阶出边邻接矩阵
    A1 = normalize_digraph(A1)  # 归一化
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))  # 组合 5 个邻接矩阵
    return A  # 返回多尺度空间图



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))  # 归一化邻接矩阵
    return A  # 返回均匀图
