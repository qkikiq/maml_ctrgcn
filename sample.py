# coding=utf-8
import numpy as np
import torch


class BatchSampler(object):
    '''
    BatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
BatchSampler：每次迭代产生一个由样本索引构成的 batch。
    每个 batch 包含 'classes_per_it' 个随机类别，
    每个类别包含 'num_support' + 'num_query' 个样本。

    __len__ 返回每个 epoch 中的迭代次数（等于 iterations）。
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the BatchSampler object 初始化函数
        Args:
        - labels: an iterable containing all the labels for the current dataset 数据集中所有样本的标签（用于推断样本索引）
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration 每次迭代中包含的类别数量
        - num_samples: number of samples for each iteration for each class (support + query) 每个类别包含的样本数量（支持集 + 查询集）
        - iterations: number of iterations (episodes) per epoch 每个 epoch 中的迭代次数
        '''
        super(BatchSampler, self).__init__()
        self.labels = labels  # 保存所有样本的标签
        # print(labels,len(labels))
        self.classes_per_it = classes_per_it # 每次迭代包含的类别数
        self.sample_per_class = num_samples # 每个类别包含的样本数（支持集 + 查询集）
        self.iterations = iterations # 每个 epoch 中的迭代次数

        # 提取所有唯一类别及每类样本数量
        self.classes, self.counts = np.unique(self.labels, return_counts=True)  #统计数据集中所有类别标签（self.classes）和它们对应的样本数量（self.counts）
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        # 所有样本的索引范围
        self.idxs = range(len(self.labels))

        # 创建一个类别数 × 每类最大样本数的矩阵，初始化为 nan（后续填入索引）
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        # 记录每个类别实际的样本数量
        self.numel_per_class = torch.zeros_like(self.classes)
        # 遍历所有样本，根据标签填入对应类别的 index 行
        for idx, label in enumerate(self.labels):
            # print((self.classes == label).numpy().astype(int))
            # 找出该类别在索引矩阵中第一个为 nan 的位置
            label_idx = np.argwhere((self.classes == label).numpy().astype(int)).item()
            # print(label_idx)
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx   # 填入样本索引
            self.numel_per_class[label_idx] += 1   # 填入样本索引

    def __iter__(self):
        '''
        yield a batch of indexes
        每次迭代返回一个 batch 的样本索引
        '''
        spc = self.sample_per_class # 每个类别的采样数量
        cpi = self.classes_per_it # 每次采样的类别数量

        for it in range(self.iterations):
            batch_size = spc * cpi  # 每个 batch 的总大小
            batch = torch.LongTensor(batch_size) # 初始化 batch 容器
            c_idxs = torch.randperm(len(self.classes))[:cpi]    # 随机选取 cpi 个类别
            # 遍历每个类别，随机选取 spc 个样本，并将其填入 batch 中
            for i, c in enumerate(self.classes[c_idxs]):
                # 定位该类别应填入 batch 的片段
                s = slice(i * spc, (i + 1) * spc)
                #  找到该类别在 self.classes 中的索引位置
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
               # 从该类别的样本中随机选取spc个样本索引
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
                # 将选出的样本索引填入 batch 的对应位置
            batch = batch[torch.randperm(len(batch))]
            # 随机打乱整个 batch 中的样本顺序
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
