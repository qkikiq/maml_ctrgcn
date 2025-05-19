import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class ntu60(Dataset):
    def __init__(self, data_path, label_path=None,  split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False,  normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
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
        #数据增强手段
        self.random_choose = random_choose  # 是否在输入序列中随机选择一部分
        self.random_shift = random_shift  # 是否随机填充序列的开始或结束位置
        self.random_move = random_move  # 是否对数据进行随机位移


        self.normalization = normalization  # 是否进行数据标准化

        self.use_mmap = use_mmap  # 是否使用内存映射模式加载数据
        self.random_rot = random_rot  # 是否执行随机旋转变换

        self.bone = bone  # 是否使用骨架数据
        self.vel = vel  # 是否使用速度信息

        # 调用加载数据的函数
        self.load_data()  # 加载数据

        if normalization:
            self.get_mean_map()

    #从 .npz 文件中读取训练或测试的骨架数据和标签
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



    #对数据进行均值和标准差计算
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


    #数据访问接口 被dataloader调用
    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

        #根据索引获取数据和标签、索引
    def __getitem__(self, index):
        # 通过索引获取样本数据和标签
        data_numpy = self.data[index]
        label = self.label[index]
        # 将 data_numpy 转换为 NumPy 数组（即使它已经是 NumPy 数组，这一步确保数据类型的一致性）
        data_numpy = np.array(data_numpy)
        # 返回处理后的数据、标签和样本的索引
        return data_numpy, label, index

# TODO: 这里需要完成对于元数据集的划分（meta_datset_train,meta_datset_test）
# class NTU60train(Dataset):
#     def __init__(self, ntu60_dataset,phase='train'):
#         """
#         :param ntu60_dataset: 已加载的 ntu60 数据集实例
#         """
#         if ntu60_dataset.split != 'train':
#             raise ValueError("传入的 ntu60 数据集必须是训练集 (split='train')")
#         # self.dataset = ntu60_dataset  # 保存传入的 ntu60 数据集实例
#         self.data = ntu60_dataset.data  # 获取训练集数据
#         self.phase= 'train'
#         # self.data = ntu60_dataset.data[0:100]  # 仅使用前100个样本进行调试
#         self.labels = ntu60_dataset.label  # 获取训练集标签
#         self.label2ind = defaultdict(list)  # 创建类别到索引的映射
#
#         # 构建类别索引字典
#         for idx, label in enumerate(self.labels):
#             self.label2ind[label].append(idx)
#
#         # 获取所有类别 ID
#         self.labelIds_base = list(self.label2ind.keys())  # 基类别 ID 列表
#         self.num_cats_base = len(self.labelIds_base)  # 基类别数量
#
#     def __len__(self):
#         """
#         返回数据集的样本数量
#         """
#         return len(self.labels)
#
#     def __getitem__(self, index):
#         """
#         根据索引返回样本数据和标签
#         :param index: 样本索引
#         :return: 样本数据和标签
#         """
#         data = torch.tensor(self.data[index], dtype=torch.float32)  # 转为张量
#         label = self.labels[index]
#         return data, label, index  # 返回数据、标签和索引

class NTU60train(Dataset):
    def __init__(self, ntu60_dataset, phase='train', meta_split_ratio=0.8, seed=42):
        """
        基于一个基础的 ntu60 训练数据集实例，创建一个用于元学习的数据集视图（元训练集或元验证集）。
        划分是基于 *类别* 进行的，而不是样本。

        :param ntu60_dataset: ntu60 类的一个实例，必须使用 split='train' 初始化。
                              该实例持有完整的原始训练数据和标签。
        :param phase: 指定创建哪个阶段的数据集视图。
                      'train': 创建元训练集。
                      'val' (或 'test'): 创建元验证/测试集。
        :param meta_split_ratio: (可选, 默认0.8) 用于元训练集的 *类别* 比例。
                                 剩余的类别将用于元验证集。
        :param seed: (可选, 默认42) 用于确定性地划分 类别的随机种子。
        """
        # --- 输入验证 ---
        if not isinstance(ntu60_dataset, ntu60):
            raise TypeError("参数 'ntu60_dataset' 必须是 ntu60 类的一个实例。")
        if ntu60_dataset.split != 'train':
            raise ValueError("传入的 'ntu60_dataset' 实例必须是用 split='train' 初始化的。")

        self.original_dataset = ntu60_dataset # 保存原始的 ntu60 数据集实例
        self.phase = phase                   # 当前阶段 ('train', 'val', 'test')
        self.meta_split_ratio = meta_split_ratio # 元训练类别比例
        self.seed = seed                     # 类别划分的随机种子

        # --- 1. 识别原始训练集中的所有唯一类别 ---
        original_labels = self.original_dataset.label # 获取原始数据集的所有标签
        # 获取唯一类别并排序，保证后续即使随机种子相同，类别列表本身顺序一致
        unique_classes = sorted(list(np.unique(original_labels)))
        num_total_classes = len(unique_classes) # 原始训练集中的总类别数

        if num_total_classes == 0:
             raise ValueError("提供的 ntu60_dataset 没有标签或类别。")

        # --- 2. 基于种子和比例，将类别划分为元训练类别集和元验证类别集 ---
        rng = random.Random(self.seed) # 使用带种子的随机数生成器，确保划分可复现
        shuffled_classes = unique_classes[:] # 创建一个副本进行打乱
        rng.shuffle(shuffled_classes)        # 原地打乱类别列表

        # 计算元训练集应包含的类别数量
        num_meta_train_classes = int(num_total_classes * self.meta_split_ratio)
        # 处理边界情况：确保至少有一个元训练类别（如果总类别>0）
        if num_meta_train_classes == 0 and num_total_classes > 0:
             num_meta_train_classes = 1
        # 处理边界情况：确保至少有一个元验证类别（如果总类别>0）
        if num_meta_train_classes == num_total_classes and num_total_classes > 0:
             num_meta_train_classes = num_total_classes - 1

        # 将打乱后的类别分配给元训练集和元验证集 (使用 set 以提高查找效率)
        self.meta_train_classes = set(shuffled_classes[:num_meta_train_classes])
        self.meta_val_classes = set(shuffled_classes[num_meta_train_classes:])

        print(f"元学习类别划分 (种子: {self.seed}):")
        print(f"  原始训练集总类别数: {num_total_classes}")
        print(f"  元训练类别 ({len(self.meta_train_classes)}): {sorted(list(self.meta_train_classes))}")
        print(f"  元验证类别 ({len(self.meta_val_classes)}): {sorted(list(self.meta_val_classes))}")


        # --- 3. 根据当前 phase 参数，筛选数据 ---
        if self.phase == 'train':
            self.target_classes = self.meta_train_classes # 目标是元训练类别
            print(f"初始化 NTU60train (phase='train')，使用 {len(self.target_classes)} 个类别。")
        elif self.phase == 'val' or self.phase == 'test': # 对类别划分而言，'val' 和 'test' 等同处理
            self.target_classes = self.meta_val_classes   # 目标是元验证类别
            print(f"初始化 NTU60train (phase='{self.phase}')，使用 {len(self.target_classes)} 个类别。")
        else:
            raise ValueError(f"无效的 phase '{self.phase}'。必须是 'train', 'val', 或 'test'。")

        # 如果当前阶段没有目标类别，进行相应处理
        if not self.target_classes:
             print(f"警告：阶段 '{self.phase}' 没有选定的类别。此数据集将为空。")
             self.data_indices = []       # 存储属于目标类别的样本在 *原始数据集* 中的索引
             self.filtered_labels = []    # 存储这些筛选后样本的标签
             self.label2ind = defaultdict(list) # 类别到 *筛选后数据集* 索引的映射
             self.labelIds_base = []      # 当前阶段（筛选后）数据集包含的类别 ID
             self.num_cats_base = 0       # 当前阶段（筛选后）数据集包含的类别数量
             return # 如果没有类别，则停止初始化

        # 筛选数据：遍历原始数据集，保留属于 target_classes 的样本
        self.data_indices = []
        self.filtered_labels = []
        for original_idx, original_label in enumerate(original_labels):
            if original_label in self.target_classes:
                self.data_indices.append(original_idx)    # 记录原始索引
                self.filtered_labels.append(original_label) # 记录对应标签

        # 如果筛选后没有样本，发出警告
        if not self.data_indices:
             print(f"警告：在阶段 '{self.phase}' 中没有找到属于目标类别的样本。此数据集将为空。")
             self.label2ind = defaultdict(list)
             self.labelIds_base = []
             self.num_cats_base = 0
             return # 如果没有样本，停止初始化

        print(f"  为阶段 '{self.phase}' 找到 {len(self.data_indices)} 个样本。")

        # --- 4. 构建筛选后数据的 label2ind 映射 ---
        # 这个映射是将 原始类别标签 映射到其样本在 *当前这个筛选后的数据集* 中的索引列表
        # 例如 label2ind[类别A] = [0, 5, 12, ...] 表示类别A的样本在此数据集中的索引是第0, 5, 12...个
        self.label2ind = defaultdict(list)
        for filtered_idx, label in enumerate(self.filtered_labels):
            self.label2ind[label].append(filtered_idx) # 使用筛选后的索引 (filtered_idx)

        # --- 5. 定义当前阶段（筛选后）数据集的基础属性 ---
        # labelIds_base 现在指的是当前这个 NTU60train 实例（特定 phase）所包含的类别
        self.labelIds_base = sorted(list(self.target_classes))
        # num_cats_base 指的是当前这个 NTU60train 实例（特定 phase）所包含的类别数量
        self.num_cats_base = len(self.labelIds_base)

    def __len__(self):
        """
        返回当前阶段（筛选后）数据集中的样本数量。
        """
        return len(self.data_indices)

    def __getitem__(self, index):
        """
        根据索引返回筛选后数据集中的样本数据和标签。

        :param index: 在 *筛选后* 数据集中的索引 (范围从 0 到 len(self)-1)。
        :return: (data_tensor, label, index)
                 data_tensor: 样本数据的 FloatTensor。
                 label: 样本的原始类别标签。
                 index: 输入的索引值 (即筛选后的索引)。
        """
        if index < 0 or index >= len(self.data_indices):
             raise IndexError(f"索引 {index} 超出范围，数据集大小为 {len(self.data_indices)}")

        # 1. 根据筛选后的索引找到在原始数据集中的索引
        original_idx = self.data_indices[index]
        # 2. 获取该样本的标签 (使用筛选后的标签列表，效率更高)
        label = self.filtered_labels[index]

        # 3. 使用原始索引从原始 ntu60_dataset 实例中获取数据
        #    原始 __getitem__ 返回 (data_numpy, label, index)，我们只需要 data_numpy
        data_numpy, _, _ = self.original_dataset[original_idx]

        # 4. 将 NumPy 数据转换为 PyTorch Tensor
        data_tensor = torch.tensor(data_numpy, dtype=torch.float32)

        # 5. 返回数据张量、标签和筛选后的索引
        return data_tensor, label, index



class FewShotDataloader():
    def __init__(self,
                 dataset,
                 nKnovel=5,  # number of novel categories.新类别数量 (K-way 中的 K)
                 nKbase=-1,  # number of base categories.基类别数量
                 nSupport=1,  # number of training examples per novel category.（每个新类别的）样本/样例数量 (N-shot 中的 N)
                 nQueryNovel=15 * 5,  # number of test examples for all the novel categories.  测试（查询集）中的新/基类别样本数量
                 nQueryBase=15 * 5,
                 batch_size=1,
                 epoch_size=1000,
                 num_workers=1,
                 ):
        self.dataset = dataset
        self.phase = self.dataset.phase


        # # 计算最大可能的新类别数量
        # # 训练阶段：新类别从基类中采样，所以最大可能数是基类总数
        # # 测试/验证阶段：新类别是预定义的新类别，最大可能数是数据集中定义的新类别总数
        # max_possible_nKnovel = (self.dataset.num_cats_base if self.phase == 'train'
        #                         else self.dataset.num_cats_novel)
        # assert (nKnovel >= 0 and nKnovel < max_possible_nKnovel)
        self.nKnovel = nKnovel

        # 获取数据集中定义的基类别总数
        max_possible_nKbase = self.dataset.num_cats_base
        # 如果 nKbase 参数为负数（通常是默认值 -1），则表示使用所有可用的基类别
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase == 'train' and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert (nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase


        self.nSupport = nSupport  # 每个新类别的支持集样本数 (N-shot)
        self.nQueryNovel = nQueryNovel  # 查询集中新类别样本总数
        self.nQueryBase = nQueryBase  # 查询集中基类别样本总数
        self.batch_size = batch_size  # 每个批次的回合数
        self.epoch_size = epoch_size  # 每个 epoch 的批次数

        self.batch_size = batch_size  # 每个批次的回合数
        self.epoch_size = epoch_size  # 每个 epoch 的批次数

        self.num_workers = num_workers  # 数据加载工作进程数
        self.is_eval_mode = (self.phase == 'test')



    def samples_cat(self, cat_id, sample_size=1):
        """
        :param cat_id: 类别ID
        :param sample_size: 每个类别的样本数
        :return: 选定类别的样本数据
        """
        assert (cat_id in self.dataset.label2ind)  # # 确保指定的类别 ID
        assert (len(self.dataset.label2ind[cat_id]) >= sample_size)   ## 确保该类别下的图像数量不少于 sample_size
        # Note: random.sample samples elements without replacement.
        # 从指定类别的样本中随机选择 sample_size 个样本
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        从指定的类别集合（基础类别或新颖类别）中随机采样指定数量的唯一类别 ID。
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.

        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.  # 指定类别集合的字符串，可以是 'base' 或 'novel'。
            sample_size: number of categories that will be sampled.  # 要采样的类别数量。

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.  # 返回一个包含唯一类别 ID 的列表，长度为 `sample_size`。
        """
        if cat_set == 'base':  # 如果类别集合是 'base'（基础类别）
            labelIds = self.dataset.labelIds_base  # 获取基础类别的 ID 列表
        elif cat_set == 'novel':  # 如果类别集合是 'novel'（新颖类别）
            labelIds = self.dataset.labelIds_novel  # 获取新颖类别的 ID 列表
        else:  # 如果类别集合既不是 'base' 也不是 'novel'
            raise ValueError('Not recognized category set {}'.format(cat_set))  # 抛出异常，提示无效的类别集合

        assert (len(labelIds) >= sample_size)  # 确保类别集合中的类别数量大于或等于要采样的数量
        # 从类别 ID 列表中随机采样指定数量的唯一类别 ID
        # 注意：random.sample 会在不放回的情况下采样元素。
        return random.sample(labelIds, sample_size)  # 返回采样的类别 ID 列表

     #从novel和base类采样指定数量的类别id
    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
       采样指定数量的基础类别（nKbase）和新类别（nKnovel）。
        categories.

        Args:
            nKbase: number of base categories    基础类别数量
            nKnovel: number of novel categories   新类别数量

        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base   包含采样的基础类别 ID 的列表，长度为 nKbase
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel   包含采样的新类别 ID 的列表，长度为 nKnovel
                categories.
        """
        if self.is_eval_mode:  # 测试阶段
            assert (nKnovel <= self.dataset.num_cats_novel)   # 确保新类别数量不超过数据集中可用的新类别数量
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))  # # 从基础类别中采样 nKbase 个类别 ID，并按顺序排序
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))  # # 从新类别中采样 nKnovel 个类别 ID，并按顺序排序
        else:  # 训练阶段
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sampleCategories('base', nKnovel + nKbase)    # 从基础类别中采样 nKnovel + nKbase 个类别 ID
            assert (len(cats_ids) == (nKnovel + nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            # 前 nKnovel 个类别作为伪新类别
            Knovel = sorted(cats_ids[:nKnovel])
            # 剩余的类别作为基础类别
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_query_examples_for_base_categories(self, Kbase, nQueryBase):
        """
        Sample `nQueryBase` number of images from the `Kbase` categories.
        从基类中采样query样本。

        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            nQueryBase: the total number of images that will be sampled.

        Returns:
            Tbase: a list of length `nQueryBase` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []   # 初始化存储采样结果的列表
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `nQueryBase`.
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nQueryBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)
            # 遍历每个类别索引及其对应的采样数量
            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                sam_ids = self.samples_cat(
                    Kbase[Kbase_idx], sample_size=NumImages)    # 从指定类别中采样指定数量的样本 ID
                Tbase += [(img_id, Kbase_idx) for img_id in sam_ids]    # 将样本ID和类别索引添加到结果列表中

        assert (len(Tbase) == nQueryBase)

        return Tbase

    def sample_support_and_query_examples_for_novel_categories(
            self, Knovel, nQueryNovel, nSupport, nKbase):
        """Samples train and test examples of the novel categories.

        Args:
            Knovel: a list with the ids of the novel categories.
            nQueryNovel: the total number of test images that will be sampled
                from all the novel categories.
            nSupport: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.

        Returns:
            Tnovel: a list of length `nQueryNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nnSupport of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:   # 如果没有新类别，直接返回空列表
            return [], []

        nKnovel = len(Knovel)   # 新类别的数量
        Tnovel = []   # 初始化存储query结果的列表
        support = []    # 初始化存储支持集结果的列表
        assert ((nQueryNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nQueryNovel / nKnovel)     # 每个新类别的测试样本数量

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.samples_cat(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nSupport))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, nKbase + Knovel_idx) for img_id in imds_tnovel]
            support += [(img_id, nKbase + Knovel_idx) for img_id in imds_ememplars]
        assert (len(Tnovel) == nQueryNovel)
        assert (len(support) == len(Knovel) * nSupport)
        random.shuffle(support)

        return Tnovel, support

        #采样训练任务
    def sample_episode(self):
        """
        采样一个训练episode
        """
        nKbase = self.nKbase  # base类数量
        nKnovel = self.nKnovel    #novel类数量
        nqueryNovel = self.nqueryNovel  #novel类query样本数量
        nqueryBase = self.nqueryBase    #base类query样本数量
        nsupport = self.nsupport  #novel类支持样本数量

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)  # 随机采样base和novel类
        Tbase = self.sample_query_examples_for_base_categories(Kbase, nqueryBase)   # 从base类别中采样query样本
        Tnovel, support = self.sample_support_and_query_examples_for_novel_categories(  # 从新类别中采样support集和query集样本
                               Knovel, nqueryNovel, nsupport, nKbase)

        # concatenate the base and novel category examples.
        query = Tbase + Tnovel   # 将基类别和新类别的query集样本合并
        random.shuffle(query)     # 随机打乱query集样本的顺序
        Kall = Kbase + Knovel     # 将基类别和新类别的类别 ID 合并

        return support, query, Kall, nKbase

    def createSupportTensorData(self, support):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        # 使用列表推导从 `examples` 中提取每个图像的索引 `img_idx`，通过 `self.dataset` 获取图像数据，并堆叠成一个张量。
        # `self.dataset[img_idx][0]` 获取图像数据，`dim=0` 表示沿第 0 维堆叠。
        sample = torch.stack(
            [self.dataset[sam_idx][0] for sam_idx, _ in support], dim=0)
            # 使用列表推导从 `examples` 中提取每个图像的类别标签 `label`，并将其转换为 `LongTensor` 类型。
        lab = torch.LongTensor([label for _, label in support])

        # 返回图像张量和标签张量。
        return sample, lab


    #数据迭代器
    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            support, query, category, nkbase = self.sample_episode()  #支持 查询 类别  基类别数量
            Xq, Yq = self.createSupportTensorData(query)   #query数据标签
            category = torch.LongTensor( category)   #类别
            if len(support) > 0:   #支持集
                Xs, Ys = self.createSupportTensorData(support)   #support数据标签
                return Xs, Ys, Xq, Yq, category, nkbase
            else:
                return Xq, Yq, category, nkbase

        #todo 创建dataset
        # 使用 `torchnet` 的 ListDataset 构造一个数据集对象，
        # 它会基于 `range(self.epoch_size)` 生成的索引序列，调用上面的 load_function 来生成每个样本
        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        # 使用 parallel 方法并行加载数据
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,  # 每个批次的样本数量
            num_workers=(0 if self.is_eval_mode else self.num_workers),  # 数据加载的工作进程数
            shuffle=(False if self.is_eval_mode else True)   # 是否对数据进行随机打乱
        )
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)




# class meta_data(Dataset):
#     def __init__(self, data_path, label_path=None, split='train', random_choose=False, random_shift=False,
#                  random_move=False, random_rot=False, normalization=False, use_mmap=False,
#                  meta_split=False, meta_split_ratio=0.2,seed=42):
#         """
#         :param meta_split: If true, split the data into meta-train and meta-test sets
#         :param meta_split_ratio: The ratio of meta-train set to meta-test set
#         """
#         self.meta_split = meta_split  # 是否进行元数据划分
#         self.meta_split_ratio = meta_split_ratio  # 元数据划分的比例
#         self.split = split   # 当前数据集的用途：'train' 或 'test'
#         self.seed = seed
#
#         # 使用已有的nt60数据集类
#         self.ntu60_data = ntu60(data_path, label_path, 'train', random_choose, random_shift, random_move, random_rot,
#                                 normalization, use_mmap)
#
#         # 只在meta_split=True时进行类别划分
#         if meta_split:
#             self.split_meta_data()
#
#     def split_meta_data(self):
#         """
#         基于训练集中划分好的样本，将其类别再拆分为 meta-train 和 meta-test
#         """
#         # 提取训练集中的所有样本的标签和数据
#         all_labels = [self.ntu60_data.label[i] for i in range(len(self.ntu60_data))]   # 获取每个样本的标签
#         all_indices = list(range(len(self.ntu60_data)))   # 获取所有样本的索引
#
#         # 获取所有类别
#         unique_labels = np.unique(all_labels)
#
#         # 设置随机种子，确保结果可复现
#         random.seed(self.seed)
#
#         # 4. 根据设定的比例选择用于 meta-train 的类别数目
#         n_train_classes = int(len(unique_labels) * self.meta_split_ratio)
#
#         train_classes = random.sample(list(unique_labels), n_train_classes)   #从所有类别中随机选出 n_train_classes 个类别用于 meta-train
#         test_classes = [cls for cls in unique_labels if cls not in train_classes]   #剩下的类别自动归为 meta-test 类别
#
#         # 按照类别划分索引（遍历所有训练数据的索引，按照类别分配到 meta-train 或 meta-test）
#         self.meta_traindata = [i for i in all_indices if self.ntu60_data.label[i] in train_classes]
#         self.meta_testdata = [i for i in all_indices if self.ntu60_data.label[i] in test_classes]
#
#     def __len__(self):
#         """
#                返回当前数据子集的长度（meta 模式下是 train/test 子集的长度，否则是完整数据集长度）
#                """
#         if self.meta_split:
#             if self.split == 'train':
#                 return len(self.meta_traindata)
#             elif self.split == 'test':
#                 return len(self.meta_testdata)
#         return len(self.ntu60_data)
#
#     def __getitem__(self, index):
#         if self.meta_split:
#             if self.split == 'train':
#                 real_index = self.meta_traindata[index]
#             elif self.split == 'test':
#                 real_index = self.meta_testdata[index]
#             else:
#                 raise NotImplementedError('Only support train/test splits in meta mode.')
#         else:
#             real_index = index
#
#         data_numpy, _, _ = self.ntu60_data[real_index]
#         label = self.ntu60_data.label[real_index]
#         return np.array(data_numpy), label