import numpy as np

from torch.utils.data import Dataset

from feeders import tools


######dataset######
class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, normalization=False, debug=False, use_mmap=False,
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

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

        #根据索引获取数据和标签
    def __getitem__(self, index):
        # 通过索引获取样本数据和标签
        data_numpy = self.data[index]
        label = self.label[index]

        # 将 data_numpy 转换为 NumPy 数组（即使它已经是 NumPy 数组，这一步确保数据类型的一致性）
        data_numpy = np.array(data_numpy)

        # 计算有效帧的数量（即在该样本中，非零的时间步数）
        # valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # 解释：
        # data_numpy.sum(0) 会计算每个时间步的所有关节的和；
        # .sum(-1) 会计算每个关节的所有轴的和（例如 x, y, z 坐标），
        # .sum(-1) 计算每个关节所有坐标轴（x, y, z）的总和，最后检查每个时间步是否有有效数据（非零）。

        # 调用工具函数 valid_crop_resize，裁剪并调整数据大小
        #valid_frame_num 表示有效帧数，p_interval 表示间隔，window_size 表示窗口大小
        #todo 样本的原始输入
        # data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        # 如果开启了随机旋转，则调用工具函数 random_rot 对数据进行旋转处理
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        # 如果启用了骨骼模式（bone），则计算骨骼数据
        if self.bone:
            from .bone_pairs import ntu_pairs  # 导入骨骼关节对（如 NTU 数据集的关节对）

            # 创建与原始数据相同形状的零矩阵
            bone_data_numpy = np.zeros_like(data_numpy)

            # 根据骨骼关节对，计算骨骼的相对位置
            for v1, v2 in ntu_pairs:
                # 用 v1 和 v2 关节之间的差值来替代原始的关节数据
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]

            # 将数据替换为骨骼数据
            data_numpy = bone_data_numpy

        # 如果启用了速度模式（vel），则计算每个关节的速度（差分）
        if self.vel:
            # 计算每个时间步的速度，数据的每一项等于当前时间步和前一个时间步的差
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            # 对最后一个时间步的速度赋值为 0（因为没有后续时间步可计算差分）
            data_numpy[:, -1] = 0

        # 返回处理后的数据、标签和样本的索引
        return data_numpy, label, index



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