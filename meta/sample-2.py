import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
import numpy as np
# Assume Feeder class is defined elsewhere or imported

class Sample(Dataset):
    def __init__(self, dataset, n_way=2, k_shot=5, q_query=5, n_episodes=100):
        """
        dataset: 一个已经实例化的 PyTorch Dataset (如 Feeder)，必须能 __getitem__ 返回 (data, label)
        n_way: 每个 episode 选择的类别数
        k_shot: 每个类别的 support 样本数
        q_query: 每个类别的 query 样本数
        n_episodes: 总共多少个 episode (这将是 Sample 数据集的长度)
        """
        self.dataset = dataset  # 使用传入的 Feeder 实例
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes

        # 不再需要调用 self.load_data()，因为 Feeder 已经加载了

        # 按 label 分类索引 (直接使用传入的 dataset)
        self.class_to_indices = defaultdict(list)
        print(f"Building class indices for Sample using dataset of length: {len(self.dataset)}") # Debug print
        for i in range(len(self.dataset)):
            # 假设 self.dataset[i] 返回 (data, label) 或 (data, label, index)
            # 我们只需要 label
            item = self.dataset[i]
            # 检查返回值的结构，假设标签是第二个元素
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                 _, label = item[0], item[1] # 或者根据 Feeder.__getitem__ 的实际返回值调整索引
                 # 如果 Feeder.__getitem__ 返回 (data_numpy, label, index)
                 # _, label, _ = item
            else:
                # 如果 Feeder.__getitem__ 只返回数据和标签，没有索引
                 raise ValueError(f"Unexpected item structure from dataset.__getitem__({i}): {item}")
                 # 或者如果确定 Feeder 返回 (data, label)
                 # _, label = item

            # 确保 label 是可哈希的类型 (通常是 int 或 str)
            if not isinstance(label, (int, str, np.integer)):
                 # 如果标签是 numpy 数组或 tensor，可能需要转换
                 try:
                     label = int(label.item()) if hasattr(label, 'item') else int(label)
                 except Exception as e:
                     raise TypeError(f"Label type {type(label)} at index {i} is not suitable for dictionary key. Error: {e}")

            self.class_to_indices[label].append(i)

        self.classes = list(self.class_to_indices.keys())
        if not self.classes:
             print("Warning: No classes found after indexing the dataset in Sample.")
        if len(self.classes) < self.n_way:
             raise ValueError(f"Not enough classes ({len(self.classes)}) in the dataset for n_way={self.n_way}")
        print(f"Found {len(self.classes)} classes in Sample.") # Debug print

    def __len__(self):
        # 长度是 episodes 的数量
        return self.n_episodes

    def __getitem__(self, index):
        # index 参数在这里实际上没有被使用，因为每个 episode 都是随机生成的
        # 但 __getitem__ 需要这个参数

        if len(self.classes) < self.n_way:
             # 再次检查，以防数据集过小
             raise ValueError(f"Cannot sample {self.n_way} classes, only {len(self.classes)} available.")

        selected_classes = random.sample(self.classes, self.n_way)
        support_data, support_labels = [], []
        query_data, query_labels = [], []

        label_map = {original_label: mapped_label for mapped_label, original_label in enumerate(selected_classes)}

        for class_label in selected_classes:
            indices = self.class_to_indices[class_label]
            # 检查是否有足够的样本
            required_samples = self.k_shot + self.q_query
            if len(indices) < required_samples:
                # 处理样本不足的情况：可以报错，或者重复采样（可能不理想）
                print(f"Warning: Class {class_label} has only {len(indices)} samples, but {required_samples} are needed. Sampling with replacement.")
                # 或者直接报错：
                # raise ValueError(f"Class {class_label} has only {len(indices)} samples, but {required_samples} are needed for k_shot={self.k_shot} and q_query={self.q_query}")
                selected_indices = random.choices(indices, k=required_samples) # 使用 choices 进行重复采样
            else:
                selected_indices = random.sample(indices, required_samples)

            support_idx = selected_indices[:self.k_shot]
            query_idx = selected_indices[self.k_shot:]

            new_label = label_map[class_label]

            for idx in support_idx:
                # 获取数据，假设 Feeder 返回 (data, label) 或 (data, label, index)
                item = self.dataset[idx]
                data = item[0] # 假设数据是第一个元素
                support_data.append(data)
                support_labels.append(new_label)

            for idx in query_idx:
                item = self.dataset[idx]
                data = item[0] # 假设数据是第一个元素
                query_data.append(data)
                query_labels.append(new_label)

        # 转为 tensor 堆叠
        # 确保数据是 numpy 数组以便 from_numpy
        try:
            # 先检查数据类型
            if isinstance(support_data[0], torch.Tensor):
                # 如果已经是 Tensor，直接堆叠
                support_data_tensor = torch.stack(support_data)
                query_data_tensor = torch.stack(query_data)
            else:
                # 如果是 NumPy 数组，转换后堆叠
                support_data_tensor = torch.stack([torch.from_numpy(x) for x in support_data])
                query_data_tensor = torch.stack([torch.from_numpy(x) for x in query_data])
        except Exception as e:
            print(f"错误类型: {type(support_data[0])}")
            print(f"数据转换错误: {e}")
            raise

        support_labels_tensor = torch.tensor(support_labels)
        query_labels_tensor = torch.tensor(query_labels)

        # 返回一个 episode
        return support_data_tensor, support_labels_tensor, query_data_tensor, query_labels_tensor, index