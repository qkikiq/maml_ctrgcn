import numpy as np
import torch
from torch.utils.data import Dataset

from feeders import tools

import random

######dataset######
class ntu60_few_shot(Dataset):
    def __init__(self, data_path, label_path=None, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False,
            # 小样本学习参数
                batchsize = 10000,
                n_way = 5,
                k_shot = 5,
                k_query = 15,
                startidx = 0,
                 ):
        """
        :param data_path:  数据路径
        :param label_path:  标签路径
        :param split: training set or test set  训练集（'train'）或测试集（'test'）
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
        self.batchsize = batchsize
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.startidx = startidx


        # 调用加载数据的函数
        self.load_data()  # 加载数据

        self.cls_num = len(self.data)  # 类别为64

        self.create_batch(self.batchsize)

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
        #为元学习创建支持集和查询集的批次
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """

        # 获取样本数量足够的类别
        valid_classes = []
        for cls in range(len(self.data)):
            if len(self.data[cls]) >= self.k_shot + self.k_query:
                valid_classes.append(cls)

        # 确保有足够的有效类别进行n_way分类
        if len(valid_classes) < self.n_way:
            raise ValueError(
                f"找不到足够的类别进行{self.n_way}-way分类。样本数大于{self.k_shot + self.k_query}的类别只有{len(valid_classes)}个。")


        self.support_x_batch = []  # 初始化支持集批次
        self.query_x_batch = []  # 初始化查询集批次
        for b in range(batchsz):  # 遍历每个批次
            # 1.select n_way classes randomly  随机选择 n_way 个类别
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # 所有类别中随机选择若干个类别  不放回  打乱类别顺序
            np.random.shuffle(selected_cls)  #打乱类别顺序
            support_x = []  # 当前批次的支持集
            query_x = []  # 当前批次的查询集
            for cls in selected_cls: # 遍历每个类别
                # 2. select k_shot + k_query for each class
                selected_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_idx)
                indexDtrain = np.array(selected_idx[:self.k_shot])  # idx for Dtrain  支持集索引
                indexDtest = np.array(selected_idx[self.k_shot:])  # idx for Dtest  查询集索引
                #获取支持集和查询集的文件名
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(
                    np.array(self.data[cls])[indexDtest].tolist())
            # 打乱支持集和查询集的顺序
            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)
            # 将当前批次的支持集和查询集添加到批次列表中
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __len__(self):
        return self.batchsize

    def __iter__(self):
        return self

    # 根据索引获取数据和标签
    def __getitem__(self, index):
        """
        根据索引获取一个元学习任务（episode）
        :param index: 批次索引，0 <= index < batchsz
        :return: 支持集数据、支持集标签、查询集数据、查询集标签
        """
        # 确保索引有效
        if index >= len(self.support_x_batch):
            raise IndexError(f"索引 {index} 超出批次范围 {len(self.support_x_batch)}")

            # 获取当前批次的支持集和查询集
        support_x_task = self.support_x_batch[index]  # 支持集，格式为[n_way, k_shot]的嵌套列表
        query_x_task = self.query_x_batch[index]  # 查询集，格式为[n_way, k_query]的嵌套列表

        # 计算支持集和查询集的大小
        n_way = len(support_x_task)  # 类别数
        setsz = n_way * self.k_shot  # 支持集总样本数
        querysz = n_way * self.k_query  # 查询集总样本数

        # 初始化数据数组
        support_data = np.zeros((setsz, *self.data[0].shape))  # 支持集数据，形状为[setsz, C, T, V, M]
        support_label = np.zeros(setsz, dtype=np.int32)  # 支持集标签
        query_data = np.zeros((querysz, *self.data[0].shape))  # 查询集数据，形状为[querysz, C, T, V, M]
        query_label = np.zeros(querysz, dtype=np.int32)  # 查询集标签

        # 展平支持集和查询集数据，并提取对应标签
        support_idx = 0  # 支持集样本计数器
        query_idx = 0  # 查询集样本计数器

        # 为每个类别生成相对标签
        for class_idx, (support_samples, query_samples) in enumerate(zip(support_x_task, query_x_task)):
            # 处理支持集样本
            for sample in support_samples:
                # 复制骨架数据
                data_numpy = np.array(sample)

                # 如果启用了随机旋转
                if self.random_rot:
                    data_numpy = tools.random_rot(data_numpy)

                # 如果启用了骨骼模式
                if self.bone:
                    from feeders.bone_pairs import ntu_pairs
                    bone_data_numpy = np.zeros_like(data_numpy)
                    for v1, v2 in ntu_pairs:
                        bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
                    data_numpy = bone_data_numpy

                    # 如果启用了速度模式
                    if self.vel:
                        data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
                        data_numpy[:, -1] = 0

                # 标准化处理
                if self.normalization:
                    data_numpy = (data_numpy - self.mean_map) / self.std_map

                # 存储处理后的样本和标签
                support_data[support_idx] = data_numpy
                support_label[support_idx] = class_idx  # 使用相对标签(0到n_way-1)
                support_idx += 1

            # 处理查询集样本
            for sample in query_samples:
                # 复制骨架数据
                data_numpy = np.array(sample)

                # 如果启用了随机旋转
                if self.random_rot:
                    data_numpy = tools.random_rot(data_numpy)

                # 如果启用了骨骼模式
                if self.bone:
                    from feeders.bone_pairs import ntu_pairs
                    bone_data_numpy = np.zeros_like(data_numpy)
                    for v1, v2 in ntu_pairs:
                        bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
                    data_numpy = bone_data_numpy

                    # 如果启用了速度模式
                    if self.vel:
                        data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
                        data_numpy[:, -1] = 0

                # 标准化处理
                if self.normalization:
                    data_numpy = (data_numpy - self.mean_map) / self.std_map

                # 存储处理后的样本和标签
                query_data[query_idx] = data_numpy
                query_label[query_idx] = class_idx  # 使用相对标签(0到n_way-1)
                query_idx += 1

        # 打乱支持集和查询集的顺序
        support_perm = np.random.permutation(setsz)
        query_perm = np.random.permutation(querysz)

        support_data = support_data[support_perm]
        support_label = support_label[support_perm]
        query_data = query_data[query_perm]
        query_label = query_label[query_perm]

        support_data = torch.FloatTensor(support_data)
        support_label = torch.LongTensor(support_label)
        query_data = torch.FloatTensor(query_data)
        query_label = torch.LongTensor(query_label)

        return support_data, support_label, query_data, query_label