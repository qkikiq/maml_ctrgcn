import random
import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split



def load_data():
    """
    模拟加载数据的函数。
    返回一个字典，key 是类别名称，value 是该类别下的样本列表 (NumPy array)。
    """
    data = defaultdict(list)
    num_classes = 60  #  60 个类别
    samples_per_class = 100   # 每个类别 100 个样本
    for i in range(num_classes):
        data['class_' + str(i)] = np.random.rand(samples_per_class, 20)  # 假设每个样本有 20 个特征
    return data


def split_meta_dataset(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    将数据集划分为元训练集、元验证集和元测试集（基于类别）。

    Args:
        data (dict): 包含类别和对应样本的数据字典。
        train_ratio (float): 元训练集类别比例。
        val_ratio (float): 元验证集类别比例。
        test_ratio (float): 元测试集类别比例。

    Returns:
        tuple: 包含元训练集、元验证集和元测试集的类别列表。
    """
    classes = list(data.keys())
    train_classes, temp_classes = train_test_split(classes, test_size=(val_ratio + test_ratio), random_state=42)
    val_classes, test_classes = train_test_split(temp_classes, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    return train_classes, val_classes, test_classes

def create_meta_task(data, classes, n_way, k_shot, query_size):
    """
    创建一个元学习任务。

    Args:
        data (dict): 包含类别和对应样本的数据字典。
        classes (list): 当前任务包含的类别列表。
        n_way (int): 任务中的类别数量（N-way）。
        k_shot (int): 每个类别在支持集中的样本数量（K-shot）。
        query_size (int): 每个类别在查询集中的样本数量。

    Returns:
        tuple: (support_set, query_set)，每个集合都是一个包含样本和标签的元组 (features, labels)。
            - support_set: (torch.Tensor, torch.Tensor)
            - query_set: (torch.Tensor, torch.Tensor)
    """
    selected_classes = random.sample(classes, n_way)
    support_samples = []
    support_labels = []
    query_samples = []
    query_labels = []
    label_map = {class_name: i for i, class_name in enumerate(selected_classes)}

    for i, class_name in enumerate(selected_classes):
        samples = random.sample(data.get(class_name, []), k_shot + query_size)
        support_samples.extend(samples[:k_shot])
        support_labels.extend([label_map.get(class_name)] * k_shot)
        query_samples.extend(samples[-query_size:])
        query_labels.extend([label_map.get(class_name)] * query_size)

    support_features = torch.tensor(np.array(support_samples), dtype=torch.float32)
    support_targets = torch.tensor(np.array(support_labels), dtype=torch.long)
    query_features = torch.tensor(np.array(query_samples), dtype=torch.float32)
    query_targets = torch.tensor(np.array(query_labels), dtype=torch.long)

    return (support_features, support_targets), (query_features, query_targets)