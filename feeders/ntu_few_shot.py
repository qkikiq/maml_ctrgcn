import numpy as np

from torch.utils.data import Dataset

from feeders import tools

import random

######dataset######
class ntu60_few_shot(Dataset):
    def __init__(self, data_path, label_path=None, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False,
            # 小样本学习参数
                batchsz=10000,
                n_way=5,
                k_shot=1,
                k_query=15,
                num_episodes=10
                 ):
        """
        :param data_path:  数据路径
        :param label_path:  标签路径
        :param split: training set or test set  训练集（'train'）或测试集（'test'）
        随机选择
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:  对输入数据应用某种形式的随机位移
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence 用于裁剪或填充输入序列到指定长度
        :param normalization: If true, normalize input sequence  标准化
        :param debug: If true, only use the first 100 samples  调试代码时减少数据量
        #使用内存映射（mmap）模式加载数据
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        骨架模态
        :param bone: use bone modality or not
        运动模态
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """
        self.debug = debug  # 设置是否启用调试模式
        self.data_path = data_path  # 存储数据路径
        self.label_path = label_path  # 存储标签路径
        self.split = split  # 数据集划分，训练集或测试集
        self.random_choose = random_choose  # 是否在输入序列中随机选择一部分
        self.random_shift = random_shift  # 是否随机填充序列的开始或结束位置
        self.random_move = random_move  # 是否对数据进行随机位移
        self.normalization = normalization  # 是否进行数据标准化
        self.use_mmap = use_mmap  # 是否使用内存映射模式加载数据
        self.random_rot = random_rot  # 是否执行随机旋转变换
        self.bone = bone  # 是否使用骨架数据
        self.vel = vel  # 是否使用速度信息

        # 小样本学习特定属性
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_episodes = num_episodes

        # 调用加载数据的函数
        self.load_data()  # 加载数据

        self.create_batch(self.batchsz)

        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        # 加载 `.npz` 格式的数据文件
        npz_data = np.load(self.data_path)
        if self.split == 'train':   # 如果是训练集
            self.data = npz_data['x_train']   # 加载训练数据 x_train
            self.label = np.where(npz_data['y_train'] > 0)[1]  # 获取训练集标签，将大于0的标签转为类别索引
            # 为每个样本创建一个名称（如 train_0, train_1, ...）
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':  # 如果是测试集
            self.data = npz_data['x_test']  # 加载测试数据 x_test
            self.label = np.where(npz_data['y_test'] > 0)[1] # 获取测试集标签，将大于0的标签转为类别索引
            # 为每个样本创建一个名称（如 test_0, test_1, ...）
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')  # 如果 split 不是 train 或 test，则抛出异常
        N, T, _ = self.data.shape  # 获取数据的维度，N 是样本数，T 是时间步长，_ 表示关节数（25个）
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data   # 获取类中的数据集（形状为 (N, C, T, V, M)）
        N, C, T, V, M = data.shape

        # 计算 mean_map：对数据在时间维度T和样本维度M上进行均值计算
        # 首先在时间轴（axis=2）上计算均值，并保持维度（keepdims=True），结果为 (N, C, 1, V, M)
        # 然后再在样本维度M上计算均值，结果为 (N, C, 1, V, 1)
        # 最后再在样本维度N上计算均值，得到每个关节、通道、维度的均值结果，维度为 (C, 1, V, 1)
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        # 计算 std_map：对数据进行转置后重塑，计算标准差
        # 将原数据转置为 (N, T, M, C, V) 并重塑为 (N * T * M, C * V)
        # 然后在轴0上计算标准差，得到每个通道和每个关节的标准差
        # 计算完标准差后，将其重塑为 (C, 1, V, 1)，对应每个通道和每个关节的标准差
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def create_batch(self, batchsz):
        """
        为元学习创建支持集和查询集的批次
        :param batchsz: 批次大小（任务数量）
        :return:
        """
        self.support_x_batch = []  # 初始化支持集批次
        self.query_x_batch = []  # 初始化查询集批次
        
        # 获取可用的类别
        available_classes = np.unique(self.label)
        
        # 确保类别数量足够
        if len(available_classes) < self.n_way:
            raise ValueError(f"可用类别数 ({len(available_classes)}) 少于 n_way ({self.n_way})。")
        
        for b in range(batchsz):  # 遍历每个批次
            # 1. 随机选择 n_way 个类别
            selected_classes = np.random.choice(available_classes, self.n_way, replace=False)  # 无重复选择
            np.random.shuffle(selected_classes)  # 打乱类别顺序
            
            support_x = []  # 当前批次的支持集
            query_x = []  # 当前批次的查询集
            
            for cls in selected_classes:  # 遍历每个选定的类别
                # 获取该类别的所有样本索引
                class_indices = np.where(self.label == cls)[0]
                
                # 确保样本数量足够
                if len(class_indices) < (self.k_shot + self.k_query):
                    continue  # 如果样本不足，跳过这个类别
                
                # 2. 为每个类别随机选择 k_shot + k_query 个样本
                selected_indices = np.random.choice(class_indices, self.k_shot + self.k_query, replace=False)
                np.random.shuffle(selected_indices)
                
                # 分割为支持集和查询集
                support_indices = selected_indices[:self.k_shot]  # 支持集索引
                query_indices = selected_indices[self.k_shot:]  # 查询集索引
                
                # 获取支持集和查询集的样本索引列表
                support_x.append(support_indices.tolist())
                query_x.append(query_indices.tolist())
            
            # 打乱支持集和查询集的顺序
            random.shuffle(support_x)
            random.shuffle(query_x)
            
            # 将当前批次的支持集和查询集添加到批次列表中
            self.support_x_batch.append(support_x)  # 添加到支持集批次
            self.query_x_batch.append(query_x)  # 添加到查询集批次

    def __len__(self):
        return self.batchsz

    def __iter__(self):
        return self


    #根据索引获取数据和标签
    def __getitem__(self, index):
        """
        根据索引获取一个元学习任务（episode）
        :param index: 批次索引，0 <= index < batchsz
        :return: 支持集数据、支持集标签、查询集数据、查询集标签
        """
        # 确保索引有效
        if index >= len(self.support_x_batch):
            raise IndexError(f"索引 {index} 超出批次范围 {len(self.support_x_batch)}")

        # 初始化支持集和查询集的数据结构
        support_x = []  # 支持集数据
        support_y = []  # 支持集标签
        query_x = []  # 查询集数据
        query_y = []  # 查询集标签


        # 处理支持集
        for i, class_indices in enumerate(self.support_x_batch[index]):
            for idx in class_indices:
                # 获取数据样本
                data = self.data[idx]
                data = np.array(data)  # 确保是numpy数组

                # 应用数据变换
                if self.random_rot:
                    data = tools.random_rot(data)
                if self.bone:
                    from .bone_pairs import ntu_pairs
                    bone_data = np.zeros_like(data)
                    for v1, v2 in ntu_pairs:
                        bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
                    data = bone_data
                if self.vel:
                    data[:, :-1] = data[:, 1:] - data[:, :-1]
                    data[:, -1] = 0

                # 如果需要标准化
                if self.normalization:
                    data = (data - self.mean_map) / self.std_map

                support_x.append(data)
                support_y.append(i)  # 使用相对类别标签（0到n_way-1）

        # 处理查询集
        for i, class_indices in enumerate(self.query_x_batch[index]):
            for idx in class_indices:
                # 获取数据样本
                data = self.data[idx]
                data = np.array(data)  # 确保是numpy数组

                # 应用与支持集相同的数据变换
                if self.random_rot:
                    data = tools.random_rot(data)
                if self.bone:
                    from .bone_pairs import ntu_pairs
                    bone_data = np.zeros_like(data)
                    for v1, v2 in ntu_pairs:
                        bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
                    data = bone_data
                if self.vel:
                    data[:, :-1] = data[:, 1:] - data[:, :-1]
                    data[:, -1] = 0

                # 如果需要标准化
                if self.normalization:
                    data = (data - self.mean_map) / self.std_map

                query_x.append(data)
                query_y.append(i)  # 使用相对类别标签

        # 将列表转换为张量
        support_x = np.stack(support_x, axis=0)
        support_y = np.array(support_y)
        query_x = np.stack(query_x, axis=0)
        query_y = np.array(query_y)
        
        return support_x, support_y, query_x, query_y



    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')  # 将输入的类名（包含模块路径）按照 '.' 分割成组件列表
    mod = __import__(components[0])  # 导入路径中的第一个模块
    for comp in components[1:]:  # 遍历剩下的组件（模块路径的其余部分）
        mod = getattr(mod, comp)  # 使用 getattr 动态地从当前模块中获取属性（子模块或类）
    return mod  # 返回最终的类或函数



















# import numpy as np
#
# import random
# from feeders.feeder_ntu import Feeder
#
#
# ######dataset######
# class ntu60_few_shot(Feeder):
#     def __init__(self, data_path, label_path=None, split='train',
#                  n_way=5, k_shot=1, q_query=1, num_tasks=1000,
#                  # 继承自 Feeder 的参数 (可以设置默认值或直接传递)
#                  random_choose=False, random_shift=False, random_move=False,
#                  random_rot=False, normalization=False, debug=False,
#                  use_mmap=False, bone=False, vel=False):
#
#         # 调用父类的 __init__ 方法加载所有数据并设置基本的转换
#         super().__init__(data_path=data_path, label_path=label_path,
#                          split=split, random_choose=random_choose, random_shift=random_shift,
#                          random_move=random_move, random_rot=random_rot,
#                          normalization=normalization, debug=debug, use_mmap=use_mmap,
#                          bone=bone, vel=vel)
#
#         self.process_single_sample = self.process_sample
#         self.n_way = n_way  # N-way 分类任务中的类别数量 N
#         self.k_shot = k_shot  # 每个类别的支持集 (support set) 样本数量 K
#         self.q_query = q_query  # 每个类别的查询集 (query set) 样本数量 Q
#         self.num_tasks = num_tasks  # 每个 epoch 生成的任务（episode）数量
#
#         # 按类别组织数据，方便进行 N-way K-shot 采样
#         self._organize_data_by_class()
#
#         if self.debug:  # 如果处于调试模式，减少任务数量
#             self.num_tasks = min(self.num_tasks, 50)
#
#     def process_sample(self, data):
#         """处理单个样本的函数，应用与 Feeder.__getitem__ 相同的处理逻辑"""
#         data = np.array(data)
#         if self.random_rot:
#             from feeders import tools  # 导入放在函数内部，避免全局导入问题
#             data = tools.random_rot(data)
#         if self.bone:
#             from .bone_pairs import ntu_pairs
#             bone_data = np.zeros_like(data)
#             for v1, v2 in ntu_pairs:
#                 bone_data[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
#             data = bone_data
#         if self.vel:
#             data[:, :-1] = data[:, 1:] - data[:, :-1]
#             data[:, -1] = 0
#         return data
#
#     def _organize_data_by_class(self):
#         self.data_by_class = {}  # 初始化一个字典来按类别存储数据
#         for i, label in enumerate(self.label):  # 遍历所有标签和对应的索引
#             if label not in self.data_by_class:
#                 self.data_by_class[label] = []  # 如果类别首次出现，创建一个空列表
#             # 存储原始数据，而不是索引，这样后续可以直接使用
#             self.data_by_class[label].append(self.data[i])  # 将数据样本添加到对应类别的列表中
#
#         # 筛选掉样本数不足 k_shot + q_query 的类别
#         self.available_classes = []  # 可用于构建任务的类别列表
#         for class_label, samples in self.data_by_class.items():
#             if len(samples) >= self.k_shot + self.q_query:
#                 self.available_classes.append(class_label)
#
#         if len(self.available_classes) < self.n_way:
#             raise ValueError(f"没有足够的类别 ({len(self.available_classes)}) 拥有至少 "
#                              f"{self.k_shot + self.q_query} 个样本来构成一个 N-way ({self.n_way}) 任务。")
#         if self.debug:
#             print(f"已按类别组织数据。找到 {len(self.available_classes)} 个适合小样本任务的类别。")
#
#     def __len__(self):
#         # 这代表了我们一个 epoch 能生成的任务（episode）数量
#         return self.num_tasks
#
#     def __getitem__(self, index):
#         # index 只是一个虚拟的任务ID，我们每次都生成一个新的随机任务
#         data_numpy = self.data[index]
#         label = self.label[index]
#
#         # 将 data_numpy 转换为 NumPy 数组（即使它已经是 NumPy 数组，这一步确保数据类型的一致性）
#         data_numpy = np.array(data_numpy)
#
#         # 如果开启了随机旋转，则调用工具函数 random_rot 对数据进行旋转处理
#         if self.random_rot:
#             data_numpy = tools.random_rot(data_numpy)
#
#         # 如果启用了骨骼模式（bone），则计算骨骼数据
#         if self.bone:
#             from .bone_pairs import ntu_pairs  # 导入骨骼关节对（如 NTU 数据集的关节对）
#
#             # 创建与原始数据相同形状的零矩阵
#             bone_data_numpy = np.zeros_like(data_numpy)
#
#             # 根据骨骼关节对，计算骨骼的相对位置
#             for v1, v2 in ntu_pairs:
#                 # 用 v1 和 v2 关节之间的差值来替代原始的关节数据
#                 bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
#
#             # 将数据替换为骨骼数据
#             data_numpy = bone_data_numpy
#
#
#             # 如果启用了速度模式（vel），则计算每个关节的速度（差分）
#             if self.vel:
#                 # 计算每个时间步的速度，数据的每一项等于当前时间步和前一个时间步的差
#                 data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
#                 # 对最后一个时间步的速度赋值为 0（因为没有后续时间步可计算差分）
#                 data_numpy[:, -1] = 0
#
#
#         # 1. 随机选择 N_WAY 个类别
#         selected_class_labels = random.sample(self.available_classes, self.n_way)
#
#         support_data_list = []
#         support_label_list = []
#         query_data_list = []
#         query_label_list = []
#
#         # 从已加载的数据获取形状信息 (N,C,T,V,M)
#         #self.process_single_sample # 返回的数据 data_numpy 期望形状为 (C,T,V,M)
#
#         for relative_label, class_label in enumerate(selected_class_labels):
#             class_samples = self.data_by_class[class_label]  # 获取当前选定类别的所有样本
#
#             # 2. 对于每个类别，采样 K_SHOT 个支持样本和 Q_QUERY 个查询样本
#             num_samples_to_pick = self.k_shot + self.q_query
#             # 从当前类别的样本中随机选择不重复的索引
#             selected_indices_in_class = random.sample(range(len(class_samples)), num_samples_to_pick)
#
#             # 处理并添加支持集样本
#             for i in range(self.k_shot):
#                 sample_data = class_samples[selected_indices_in_class[i]]
#                 # 对样本数据进行处理（增强、标准化等）
#                 # 使用 .copy() 以避免对原始数据进行意外的就地修改
#                 processed_sample = self.process_single_sample(sample_data.copy())
#                 support_data_list.append(processed_sample)
#                 support_label_list.append(relative_label)  # 添加相对标签 (0 到 N_WAY-1)
#
#             # 处理并添加查询集样本
#             for i in range(self.q_query):
#                 sample_data = class_samples[selected_indices_in_class[self.k_shot + i]]
#                 processed_sample = self.process_single_sample(sample_data.copy())
#                 query_data_list.append(processed_sample)
#                 query_label_list.append(relative_label)
#
#         # 将列表堆叠成 NumPy 数组
#         # 支持集数据形状: (N_WAY * K_SHOT, C, T, V, M)
#         # 查询集数据形状: (N_WAY * Q_QUERY, C, T, V, M)
#         support_data = np.stack(support_data_list, axis=0)
#         support_labels = np.array(support_label_list, dtype=np.int64)
#         query_data = np.stack(query_data_list, axis=0)
#         query_labels = np.array(query_label_list, dtype=np.int64)
#
#         # 你可能想进一步重塑它们，例如，对于原型网络 (prototypical networks)：
#         # 支持集: (N_WAY, K_SHOT, C, T, V, M)
#         # 查询集: (N_WAY * Q_QUERY, C, T, V, M) 或 (N_WAY, Q_QUERY, C, T, V, M)
#
#         # 当前输出 (扁平化的 N*K, N*Q 格式):
#         return support_data, support_labels, query_data, query_labels