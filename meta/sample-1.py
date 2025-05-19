import numpy as np
import random
from collections import defaultdict
import torch

from torch.utils.data import Dataset

class Sample(Dataset):
    def __init__(self,dataset, n_way=2, k_shot=5, q_query=5, n_episodes=100):
        """
        dataset: 一个 PyTorch Dataset，必须能 __getitem__ 返回 (data, label)
        n_way: 每个 episode 选择的类别数
        k_shot: 每个类别的 support 样本数
        q_query: 每个类别的 query 样本数
        n_episodes: 总共多少个 episode
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes
        # 调用加载数据的函数
        self.load_data()  # 加载数据

        # 按 label 分类索引
        self.class_to_indices = defaultdict(list)
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            self.class_to_indices[label].append(i)
        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, index):
        selected_classes = random.sample(self.classes, self.n_way)  # 从所有类别中随机选择 n_way 个类别
        # 初始化支持集（support）和查询集（query）的数据和标签列表
        support_data, support_labels = [], []
        query_data, query_labels = [], []

        # 初始化支持集（support）和查询集（query）的数据和标签列表
        label_map = {label: idx for idx, label in enumerate(selected_classes)}  # 重新编码标签从 0 开始

        # 遍历每个选中的类别
        for class_label in selected_classes:
            indices = self.class_to_indices[class_label]  # 获取当前类别对应的样本索引列表
            selected_indices = random.sample(indices, self.k_shot + self.q_query)   # 从当前类别中随机选择 k_shot + q_query 个样本索引
            support_idx = selected_indices[:self.k_shot]   # 从当前类别中随机选择 k_shot + q_query 个样本索引
            query_idx = selected_indices[self.k_shot:]  # 后 q_query 个样本作为查询集

            # 遍历支持集的索引
            for idx in support_idx:
                data, label = self.dataset[idx]  # 根据索引获取数据和标签
                support_data.append(data) # 将数据添加到支持集数据列表
                support_labels.append(label_map[label])  # 将标签映射为新的索引后添加到支持集标签列表

            # 遍历查询集的索引
            for idx in query_idx:
                data, label = self.dataset[idx]  # 遍历查询集的索引
                query_data.append(data)   # 将数据添加到查询集数据列表
                query_labels.append(label_map[label])   # 将数据添加到查询集数据列表

        # 转为 tensor  堆叠
        support_data = torch.stack([torch.from_numpy(x) for x in support_data])
        support_labels = torch.tensor(support_labels)
        query_data = torch.stack([torch.from_numpy(x) for x in query_data])
        query_labels = torch.tensor(query_labels)

        return support_data, support_labels, query_data, query_labels


