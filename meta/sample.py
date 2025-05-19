import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
import numpy as np

# 假设 Feeder 类已经定义或导入
# from feeder import Feeder # 示例

class Sample(Dataset):
    def __init__(self, dataset, n_way=2, k_shot=5, q_query=5, n_episodes=100):
        """
        dataset: 一个已经实例化的 PyTorch Dataset (如 Feeder)，
                 其 __getitem__ 必须返回 (data, label) 或 (data, label, index)
                 其中 data 可以是 np.ndarray 或 torch.Tensor
        n_way: 每个 episode 选择的类别数
        k_shot: 每个类别的 support 样本数
        q_query: 每个类别的 query 样本数
        n_episodes: 总共多少个 episode (这将是 Sample 数据集的长度)
        """
        # ... (init 部分代码保持不变) ...
        if not isinstance(dataset, Dataset):
             raise TypeError("dataset 参数必须是一个实例化的 PyTorch Dataset")
        if len(dataset) == 0:
            raise ValueError("传入的 dataset 为空")

        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes

        self.class_to_indices = defaultdict(list)
        print(f"Sample: Building class indices from dataset with {len(self.dataset)} items...")
        found_items = 0
        for i in range(len(self.dataset)):
            try:
                item = self.dataset[i]
                if not (isinstance(item, (tuple, list)) and len(item) >= 2):
                     raise TypeError(f"Dataset item at index {i} has unexpected format: {item}")
                label = item[1]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if not isinstance(label, (int, np.integer)):
                     try:
                         label = int(label)
                     except Exception as e:
                         raise TypeError(f"Label type {type(label)} at index {i} cannot be converted to int. Error: {e}")
                self.class_to_indices[label].append(i)
                found_items += 1
            except Exception as e:
                print(f"Sample: Error processing dataset item at index {i}: {e}")
                continue

        if found_items == 0:
            raise RuntimeError("Sample: Failed to process any items from the dataset.")

        self.classes = sorted(list(self.class_to_indices.keys()))
        print(f"Sample: Found {len(self.classes)} classes. Items indexed: {found_items}.")

        if not self.classes:
             raise ValueError("Sample: No classes found after indexing the dataset.")
        if len(self.classes) < self.n_way:
             raise ValueError(f"Sample: Not enough classes ({len(self.classes)}) in the dataset for n_way={self.n_way}")

        min_samples_needed = self.k_shot + self.q_query
        for c in self.classes:
            if len(self.class_to_indices[c]) < min_samples_needed:
                print(f"Warning: Class {c} has only {len(self.class_to_indices[c])} samples, "
                      f"which is less than required k_shot+q_query ({min_samples_needed}). "
                      f"random.sample will fail if this class is selected.")


    def __len__(self):
        return self.n_episodes

    def __getitem__(self, index):
        # ... (代码 1-4 部分保持不变) ...
        # 1. 选择 n_way 个类别
        try:
            selected_classes = random.sample(self.classes, self.n_way)
        except ValueError as e:
             raise ValueError(f"Cannot sample {self.n_way} classes from {len(self.classes)} available classes. Error: {e}")

        support_data, support_labels = [], []
        query_data, query_labels = [], []

        # 2. 创建新标签映射 (0 到 n_way-1)
        label_map = {original_label: mapped_label for mapped_label, original_label in enumerate(selected_classes)}

        # 3. 为每个选中的类别采样 support 和 query 样本
        for original_label in selected_classes:
            indices = self.class_to_indices[original_label]
            try:
                selected_indices = random.sample(indices, self.k_shot + self.q_query)
            except ValueError as e:
                print(f"Error sampling for class {original_label}: It only has {len(indices)} samples, "
                      f"but tried to sample {self.k_shot + self.q_query}.")
                raise e

            support_idx = selected_indices[:self.k_shot]
            query_idx = selected_indices[self.k_shot:]
            new_label = label_map[original_label]

            # 4. 获取数据并添加到列表
            for idx in support_idx:
                try:
                    item = self.dataset[idx]
                    data = item[0] # data can be np.ndarray or torch.Tensor
                    support_data.append(data)
                    support_labels.append(new_label)
                except Exception as e:
                    print(f"Error retrieving/processing support data for index {idx}: {e}")
                    raise e

            for idx in query_idx:
                try:
                    item = self.dataset[idx]
                    data = item[0] # data can be np.ndarray or torch.Tensor
                    query_data.append(data)
                    query_labels.append(new_label)
                except Exception as e:
                    print(f"Error retrieving/processing query data for index {idx}: {e}")
                    raise e

        # 5. 转换为 Tensor 并堆叠 (带类型判断)
        try:
            if not support_data or not query_data:
                raise ValueError("Support or Query data list is empty before stacking.")

            # --- 新增逻辑判断 ---
            # 检查第一个 support data item 的类型来决定如何处理
            first_support_item = support_data[0]
            if isinstance(first_support_item, torch.Tensor):
                # 如果已经是 Tensor，直接堆叠
                support_data_tensor = torch.stack(support_data)
            elif isinstance(first_support_item, np.ndarray):
                # 如果是 NumPy 数组，使用 from_numpy 转换后堆叠
                support_data_tensor = torch.stack([torch.from_numpy(x) for x in support_data])
            else:
                # 处理未知或不支持的类型
                raise TypeError(f"Unsupported data type in support_data: {type(first_support_item)}. Expected torch.Tensor or np.ndarray.")

            # 对 query data 做同样的处理 (假设 query data 类型与 support data 一致)
            first_query_item = query_data[0]
            if isinstance(first_query_item, torch.Tensor):
                query_data_tensor = torch.stack(query_data)
            elif isinstance(first_query_item, np.ndarray):
                query_data_tensor = torch.stack([torch.from_numpy(x) for x in query_data])
            else:
                # 如果需要处理 query 和 support 类型不同的情况，可以单独判断
                # 但通常一个 dataset 返回的类型是一致的
                 raise TypeError(f"Unsupported data type in query_data: {type(first_query_item)}. Expected torch.Tensor or np.ndarray.")
            # --- 逻辑判断结束 ---

        except Exception as e:
            print(f"Error during tensor conversion/stacking: {e}")
            # 打印第一个元素的类型帮助调试 (如果列表不空)
            if support_data: print(f"Type of first support data item: {type(support_data[0])}")
            if query_data: print(f"Type of first query data item: {type(query_data[0])}")
            raise e

        support_labels_tensor = torch.tensor(support_labels, dtype=torch.long)
        query_labels_tensor = torch.tensor(query_labels, dtype=torch.long)

        # 返回一个完整的 episode 和原始请求的索引
        return support_data_tensor, support_labels_tensor, query_data_tensor, query_labels_tensor, index