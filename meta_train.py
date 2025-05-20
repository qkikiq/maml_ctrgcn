#!/usr/bin/env python
from __future__ import print_function


# 导入标准库
import argparse  # 用于解析命令行参数
import inspect  # 用于获取对象信息（例如，获取模型类的源文件）
import os  # 用于操作系统相关功能（路径操作、创建目录等）
import pickle  # 用于序列化和反序列化 Python 对象（例如，保存/加载分数）
import random  # 用于生成随机数
import shutil  # 用于文件操作（例如，复制文件、删除目录）
import sys  # 用于与 Python 解释器交互（例如，获取模块、退出程序）
import time  # 用于时间相关功能（例如，计时、获取当前时间）
from collections import OrderedDict  # 用于创建有序字典（例如，加载模型权重时保持顺序）
import traceback  # 用于获取和格式化异常信息

# 导入第三方库
from sklearn.metrics import confusion_matrix  # 用于计算混淆矩阵
import csv  # 用于读写 CSV 文件
import numpy as np  # 用于数值计算
import glob  # 用于查找符合特定规则的文件路径名

# 导入 PyTorch 相关库
import torch  # PyTorch 核心库
import torch.backends.cudnn as cudnn  # CuDNN 后端，用于加速 GPU 计算
import torch.nn as nn  # PyTorch 神经网络模块
import torch.optim as optim  # PyTorch 优化器模块
import yaml  # 用于读写 YAML 配置文件
from tensorboardX import SummaryWriter  # 用于将数据写入 TensorBoard 进行可视化
from tqdm import tqdm  # 用于显示进度条
from maml.Meta import Meta
from maml.create_adj import Adj_Matrix
# 导入自定义库
from torchlight import DictAction  # 自定义的 argparse Action，用于将 key=value 形式的参数解析为字典

# 导入并设置系统资源限制（增加打开文件的最大数量）
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)  # 获取当前进程能打开的最大文件数限制
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))  # 设置新的最大文件数限制为 2048（或原上限，取较小者）

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 剩余的代码...

# 定义一个函数来初始化随机种子，以确保实验可复现性
def init_seed(seed):
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置随机种子
    torch.manual_seed(seed)  # 为 CPU 设置随机种子
    np.random.seed(seed)  # 为 numpy 设置随机种子
    random.seed(seed)  # 为 Python 内置 random 模块设置随机种子
    # torch.backends.cudnn.enabled = False # 如果需要完全确定性，可以禁用 CuDNN（可能会牺牲性能）
    torch.backends.cudnn.deterministic = True  # 让 CuDNN 使用确定性算法（如果可用）
    torch.backends.cudnn.benchmark = False  # 禁用 CuDNN benchmark 模式（该模式会根据输入大小选择最快算法，但可能导致不确定性）

# 定义一个函数，根据字符串动态导入类
def import_class(import_str):
    """根据给定的字符串导入类。例如 'feeders.feeder.Feeder'"""
    mod_str, _sep, class_str = import_str.rpartition('.')  # 从右侧分割字符串，获取模块路径和类名
    __import__(mod_str)  # 动态导入模块
    try:
        # 从已导入的模块中获取类对象
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        # 如果找不到类，则抛出 ImportError
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

# 定义一个函数，将字符串转换为布尔值
def str2bool(v):
    """将常见的表示真/假的字符串转换为布尔值"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # 如果输入不是可识别的布尔值字符串，则抛出参数类型错误
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# 定义一个函数来创建和配置命令行参数解析器
def get_parser():
    # 参数优先级: 命令行 > 配置文件 > 默认值
    parser = argparse.ArgumentParser(
        description='few shot action recognition')  # 创建解析器对象，并设置描述信息
    # 工作目录参数
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',  # 默认工作目录
        help='the work folder for storing results')  # 参数说明
    # 配置文件路径参数
    parser.add_argument(
        '--config',
        default='./config/Small_1shot_v1.yaml',  # 默认配置文件路径
        help='path to the configuration file')  # 参数说明
    parser.add_argument(
        '--save-score',
        type=str2bool,  # 使用自定义的 str2bool 函数处理输入
        default=False,
        help='if ture, the classification score will be stored')  # 是否保存模型输出的分数
    # visulize and debug
    # === 可视化与调试参数 ===
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')  # 随机种子

    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')  # 打印日志的迭代间隔
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#epoch)')  # 保存模型的轮次间隔
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#epoch)')  # 开始保存模型的起始轮次
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#epoch)')  # 评估模型的轮次间隔
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')  # 是否打印日志到文件
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],  # 默认显示 Top-1 和 Top-5 准确率
        nargs='+',  # 允许接受一个或多个值
        help='which Top K accuracy will be shown')  # 显示哪些 Top-K 准确率

    # # feeder
    # === 数据加载器 (Feeder) 相关参数 ===
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')  # 指定要使用的数据加载器类（字符串形式）
    parser.add_argument(
        '--num-worker',
        type=int,
        default=0,  # 默认使用 32 个工作进程加载数据
        help='the number of worker for data loader')  # 数据加载器的工作进程数
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,  # 使用自定义 Action 将 key=value 参数解析为字典
        default=dict(),  # 默认为空字典
        help='the arguments of data loader for training')  # 训练数据加载器的参数
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')  # 测试数据加载器的参数

    # model
    parser.add_argument('--model', default=None, help='the model will be used')  # 指定要使用的模型类（字符串形式）
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')  # 模型的参数
    # parser.add_argument(
    #     '--weights',
    #     default=None,
    #     help='the weights for network initialization')  # 用于网络初始化的预训练权重文件路径
    # parser.add_argument(
    #     '--ignore-weights',
    #     type=str,
    #     default=[],
    #     nargs='+',
    #     help='the name of weights which will be ignored in the initialization')  # 初始化时要忽略的权重名称（部分匹配）

    # # optim
    # # === 优化器 (Optimizer) 与训练策略相关参数 ===
    # parser.add_argument(
    #     '--base-lr', type=float, default=0.01, help='initial learning rate')  # 初始学习率
    # parser.add_argument(
    #     '--step',
    #     type=int,
    #     default=[20, 40, 60],  # 默认在第 20, 40, 60 轮降低学习率
    #     nargs='+',
    #     help='the epoch where optimizer reduce the learning rate')  # 降低学习率的轮次
    parser.add_argument(
        '--device',
        type=int,
        default=0,  # 默认使用 GPU 0
        nargs='+',
        help='the indexes of GPUs for training or testing')  # 用于训练或测试的 GPU 索引
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')  # 优化器类型（如 SGD, Adam）
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')  # 是否使用 Nesterov 动量（仅 SGD）
    # parser.add_argument(
    #     '--batch-size', type=int, default=100, help='training batch size')  # 训练批次大小
    # parser.add_argument(
    #     '--test-batch-size', type=int, default=100, help='test batch size')  # 测试批次大小
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')  # 开始训练的轮次（用于断点续训）
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')  # 训练的总轮次
    # 运行模式参数
    parser.add_argument('-phase', '--phase', type=str, help='运行模式(train/test)', default='train')

    # MAML核心参数
    parser.add_argument('--n_way', type=int, help='元学习任务中的类别数量', default=5)
    parser.add_argument('--k_spt', type=int, help='支持集中每类样本数', default=1)
    parser.add_argument('--k_qry', type=int, help='查询集中每类样本数', default=15)
    parser.add_argument('--task_num', type=int, help='每个元批次的任务数', default=4)
    parser.add_argument('--meta_lr', type=float, help='元学习外层学习率', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='任务内层学习率', default=0.01)
    parser.add_argument('--update_step', type=int, help='任务内层更新步数', default=5)
    parser.add_argument('--update_step_test', type=int, help='测试时的内层更新步数', default=10)

    # 元训练控制参数
    parser.add_argument('--num_meta_epoch', type=int, help='元训练的总轮数', default=100)
    parser.add_argument('--start_meta_epoch', type=int, help='开始元训练的轮次', default=0)

    # 批量采样器参数 - 这些参数与n_way/k_spt/k_qry有重叠，建议统一
    parser.add_argument('--train_iterations', type=int, help='每个epoch的训练迭代次数', default=100)
    parser.add_argument('--test_iterations', type=int, help='测试时的迭代次数', default=100)

    # 其余不常用或冗余参数已移除
    return parser

class Processor():
    def __init__(self, arg):
        self.arg = arg  # 保存传入的参数对象
        self.save_arg()  # 调用方法保存参数到文件

        # 初始化TensorBoard写入器
        if self.arg.phase == 'train':
            self.train_writer = SummaryWriter(
                os.path.join(self.arg.work_dir, 'train'), 'train')
            self.test_writer = SummaryWriter(
                os.path.join(self.arg.work_dir, 'test'), 'test')

        # 加载模型
        self.load_model()

        # load_data(self, data_list, phase):
        self.load_data()

        # 设置输出设备
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device

        self.global_step = 0  # 初始化全局步数（用于 TensorBoard 记录）

        # 将模型移动到指定的输出设备（通常是主 GPU）
        self.model = self.model.cuda(self.output_device)

        # 如果指定了多个 GPU，则使用 DataParallel 进行数据并行
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,  # 指定使用的 GPU ID 列表
                    output_device=self.output_device)  # 指定结果聚合到的设备


    def save_arg(self):
        # 将参数对象的属性转换为字典
        arg_dict = vars(self.arg)
        # 如果工作目录不存在，则创建它
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        # 将配置写入 YAML 文件
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            # 写入执行命令
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            # 使用 YAML 格式转储参数字典
            yaml.dump(arg_dict, f)

        # 调整学习率的方法

    # 打印日志的方法
    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    # def init_dataset(self, data_list, phase):
    #     debug = False
    #     dataset = Feeder(phase=phase, data_list=data_list, debug=debug)
    #     n_classes = len(np.unique(dataset.label))
    #     if n_classes < arg.classes_per_it_tr or n_classes < arg.classes_per_it_val:
    #         raise(Exception('There are not enough classes in the dataset in order ' +
    #                         'to satisfy the chosen classes_per_it. Decrease the ' +
    #                         'classes_per_it_{tr/val} option and try again.'))
    #     return dataset

    # def init_sampler(self, labels, mode):
    #     if 'train' in mode:
    #         classes_per_it = arg.classes_per_it_tr  # 设置训练时每次episode迭代采样的类别数  5
    #         num_samples = arg.num_support_tr + arg.num_query_tr  # 11  #每个类别采样的总样本数（支持集 + 查询集）
    #         iters = arg.train_iterations  # 训练阶段每个 epoch 中的迭代（episode）次数  1000
    #     else:
    #         classes_per_it = arg.classes_per_it_val
    #         num_samples = arg.num_support_val + arg.num_query_val
    #         iters = arg.test_iterations
    #
    #     return BatchSampler(labels=labels,
    #                                     classes_per_it=classes_per_it,
    #                                     num_samples=num_samples,
    #                                     iterations=iters)

    def load_data(self):
        """
        加载训练或测试数据
        根据配置参数初始化数据加载器
        """
        # 导入数据加载器类
        Feeder = import_class(self.arg.feeder)

        # 初始化一个空字典用于存储训练和测试的数据加载器
        self.data_loader = dict()

        # 根据运行阶段加载相应数据
        if self.arg.phase == 'train':
            # 初始化训练数据集对象
            dataset = Feeder(**self.arg.train_feeder_args)

            # 初始化采样器，正确传递参数
            # sampler = self.init_sampler(dataset.label, 'train')

            # 使用PyTorch的DataLoader构造数据加载器
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.arg.task_num,  # 设置批次大小
                # batch_sampler=sampler,
                shuffle = True,  # 随机打乱数据
                num_workers=self.arg.num_worker
            )

        elif self.arg.phase == 'test':
            # 初始化测试数据集对象
            dataset = Feeder(**self.arg.test_feeder_args)

            # 初始化采样器，正确传递参数
            # sampler = self.init_sampler(dataset.label, 'test')

            # 构造测试数据加载器
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.arg.task_num,
                # batch_sampler=sampler,
                num_workers=self.arg.num_worker
            )

    def load_model(self):
        # 创建Meta模型实例
        self.model = Meta(
            args=self.arg
        )


    #外循环
    def meta_train_epoch(self, epoch, save_model_flag=False):
        """
        执行一个完整的元训练周期

        Args:
            epoch: 当前元训练轮次
            save_model_flag: 是否保存模型的标志
        """
        self.model.train()  # 设置模型为训练模式

        # 初始化指标跟踪器
        loss_values = []
        acc_values = []  # 添加准确率跟踪

        # 创建进度条以显示训练进度
        progress_bar = tqdm(self.data_loader['train'], desc=f'Meta-Train Epoch {epoch + 1}')

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(progress_bar):

            # 将数据移到 GPU
            x_spt, y_spt = x_spt.cuda(self.output_device), y_spt.cuda(self.output_device)
            x_qry, y_qry = x_qry.cuda(self.output_device), y_qry.cuda(self.output_device)

            # # 生成邻接矩阵
            # adj_spt = self.generate_adjacency_matrix(x_spt).cuda(self.output_device)
            # adj_qry = self.generate_adjacency_matrix(x_qry).cuda(self.output_device)

            # 执行元学习训练步骤
            try:
                # loss, learned_adj = self.model(x_spt, y_spt, adj_spt, x_qry, y_qry, adj_qry)
                loss, learned_adj = self.model(x_spt, y_spt, x_qry, y_qry)

                # 这里loss是查询集上的平均损失
                # 由于Meta类中已经优化过参数，所以我们只需记录损失和准确率
            except Exception as e:
                # 捕获并记录错误
                self.print_log(f"训练步骤 {step} 出错: {str(e)}")
                import traceback
                self.print_log(traceback.format_exc())
                # 跳过当前批次
                continue
            
            # 可以选择保存学习后的邻接矩阵或用于下一批次
            # 例如，每隔一定步数保存一次
            if step % self.arg.save_interval == 0:
                adj_path = os.path.join(
                    self.arg.work_dir,
                    f'learned_adj-epoch{epoch+1}-step{step}.pt'
                )
                torch.save(learned_adj, adj_path)
                self.print_log(f'保存学习后的邻接矩阵到: {adj_path}')
            
            # 收集损失
            loss_values.append(loss.item())
            
            # 更新进度条信息
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })

            # 每隔log_interval步打印一次详细信息
            if step % self.arg.log_interval == 0:
                self.print_log(
                    f'Meta-Train Epoch: {epoch + 1} [{step}/{len(self.data_loader["train"])}]\t'
                    f'Loss: {loss.item():.4f}'
                )

        # 计算整个周期的平均损失
        mean_loss = np.mean(loss_values)

        # 打印训练周期的总结
        self.print_log(
            f'\nMeta-Train Epoch: {epoch + 1} completed\t'
            f'Average Loss: {mean_loss:.4f}'
        )

        # 如果需要保存模型 - 修改保存路径，不再使用 acc
        if save_model_flag:
            model_path = os.path.join(
                self.arg.work_dir,
                f'model-epoch{epoch + 1}.pt'
            )

            # 根据不同的模型类型（并行或单GPU）保存模型
            if isinstance(self.model, nn.DataParallel):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            torch.save(state_dict, model_path)
            self.print_log(f'保存模型到: {model_path}')

        # # 每个训练周期结束时执行评估
        # if (epoch + 1) % self.arg.eval_interval == 0 or (epoch + 1 == self.arg.num_meta_epoch):
        #     self.print_log(f"\n在元训练周期 {epoch + 1} 后执行评估...")
        #     with torch.no_grad():
        #         self.meta_eval(epoch)

    def generate_adjacency_matrix(self, data):
        """
        生成格式为 [3, v, v] 的邻接矩阵
        """
        task_num, sample_num, _, _, v, _ = data.size()
        # 获取数据形状信息
        if len(data.shape) == 6:  # [task_num, sample_num, C, T, V, M]
            task_num, sample_num, _, _, num_joints, _ = data.shape
        else:
            raise ValueError(f"意外的数据形状: {data.shape}")
        
        # 创建三个子集的邻接矩阵
        A = torch.zeros((3, num_joints, num_joints), dtype=torch.float32)
        
        # 子集1: 直接连接的骨骼关节
        ntu_pairs = [
            (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
            (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
            (12, 0), (13, 12), (14, 13), (15, 14), (16, 0),
            (17, 16), (18, 17), (19, 18), (21, 22), (22, 7),
            (23, 24), (24, 11)
        ]
        
        # 构建子集1: 直接连接
        for v1, v2 in ntu_pairs:
            A[0, v1, v2] = 1
            A[0, v2, v1] = 1  # 邻接矩阵是对称的
        
        # 构建子集2: 距离为2的连接
        A[1] = torch.matmul(A[0], A[0])
        A[1] = torch.clamp(A[1], 0, 1)
        A[1] = A[1] - A[0]  # 减去直接连接的部分
        
        # 构建子集3: 距离为3的连接
        A[2] = torch.matmul(A[1], A[0])
        A[2] = torch.clamp(A[2], 0, 1)
        A[2] = A[2] - A[1] - A[0]  # 减去之前的连接
        
        # 添加自连接
        for k in range(3):
            for i in range(num_joints):
                A[k, i, i] = 1
        
        # 为每个任务和样本复制同样的邻接矩阵
        # 返回的形状是 [task_num, sample_num, 3, V, V]
        A_expanded = A.unsqueeze(0).unsqueeze(0).repeat(task_num, sample_num, 1, 1, 1)
        
        return A_expanded

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Meta-Training Phase')
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            # Global step is more about meta-updates here
            # self.global_step = self.arg.start_meta_epoch * len(self.data_loader['train']) / self.arg.tasks_per_meta_batch
            self.global_step = self.arg.start_meta_epoch * len(self.data_loader['train']) / self.arg.task_num


            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Meta-Model Parameters: {count_parameters(self.model)}')

            #开始训练
            for meta_epoch in range(self.arg.start_meta_epoch, self.arg.num_meta_epoch):
                self.print_log(f"\nStarting Meta-Epoch {meta_epoch + 1}/{self.arg.num_meta_epoch}")
                save_model_flag = (((meta_epoch + 1) % self.arg.save_interval == 0) or \
                                   (meta_epoch + 1 == self.arg.num_meta_epoch)) and \
                                  (meta_epoch + 1) > self.arg.save_epoch

                self.meta_train_epoch(meta_epoch, save_model_flag=save_model_flag)

                # # 每隔eval_interval轮评估一次模型
                # if (meta_epoch + 1) % self.arg.eval_interval == 0 or (meta_epoch + 1 == self.arg.num_meta_epoch):
                #     self.print_log(f"\n在元训练周期 {meta_epoch + 1} 后执行评估...")
                #     with torch.no_grad():
                #         loss, acc = self.meta_eval(meta_epoch)

                    # 保存最佳模型
                    # if acc > self.best_meta_acc:
                    #     self.best_meta_acc = acc
                    #     self.best_meta_acc_epoch = meta_epoch + 1
                    #     best_model_path = os.path.join(self.arg.work_dir, 'best_model.pt')
                    #
                    #     # 根据不同的模型类型（并行或单GPU）保存模型
                    #     if isinstance(self.model, nn.DataParallel):
                    #         state_dict = self.model.module.state_dict()
                    #     else:
                    #         state_dict = self.model.state_dict()

                        # torch.save(state_dict, best_model_path)
                        # self.print_log(f'保存最佳模型到: {best_model_path}, 准确率: {acc:.4f}')

                # 训练结束后的总结
            self.print_log("\n===== Meta-Training Summary =====")
            # self.print_log(f'Best Meta-Test Accuracy: {self.best_meta_acc:.4f}')
            # self.print_log(f'Best Meta-Epoch number: {self.best_meta_acc_epoch}')
            self.print_log(f'Meta-Model saved in: {self.arg.work_dir}')
            self.print_log(f'Meta LR: {self.arg.meta_lr}, Update LR: {self.arg.update_lr}')
            self.print_log(f'Num Update Steps: {self.arg.update_step}')
            self.print_log(f'Tasks per Batch: {self.arg.task_num}')
            self.print_log(f'N-way: {self.arg.n_way}, K-shot Support: {self.arg.k_spt}, K-shot Query: {self.arg.k_qry}')
            self.print_log(f'Seed: {self.arg.seed}')

            # 关闭TensorBoard写入器
            if hasattr(self, 'train_writer'):
                self.train_writer.close()
            if hasattr(self, 'test_writer'):
                self.test_writer.close()

        # elif self.arg.phase == 'test':
        #     self.print_log('Meta-Testing Phase')
        #     if not hasattr(self.arg, 'weights') or self.arg.weights is None:
        #         raise ValueError('请为元测试指定--weights参数')
        #
        #     # 加载预训练的权重
        #     self.print_log(f'从{self.arg.weights}加载元模型权重')
        #     weights = torch.load(self.arg.weights, map_location=self.output_device)
        #
        #     # 处理可能的'module.'前缀（如果是从DataParallel保存的）
        #     new_weights = OrderedDict()
        #     is_parallel = isinstance(self.model, nn.DataParallel)
        #     for k, v in weights.items():
        #         if is_parallel:
        #             if not k.startswith('module.'):
        #                 new_weights['module.' + k] = v
        #             else:
        #                 new_weights[k] = v
        #         else:
        #             if k.startswith('module.'):
        #                 new_weights[k[len('module.'):]] = v
        #             else:
        #                 new_weights[k] = v
        #
        #     self.model.load_state_dict(new_weights)
        #
        #     # 设置输出文件
        #     result_file = os.path.join(self.arg.work_dir, 'test_results.txt')
        #
        #     # 执行元测试
        #     self.print_log('开始在测试集上评估...')
        #     with torch.no_grad():
        #         loss, acc = self.meta_eval(0)  # 使用0作为epoch参数，因为只是测试
        #
        #     self.print_log(f'测试损失: {loss:.4f}, 测试准确率: {acc:.4f}')
        #     self.print_log('元测试完成。\n')

        return


if __name__ == '__main__':
    # 调用 get_parser 函数创建一个命令行参数解析器 parser。
    parser = get_parser()

    # load arg form config file
    # 解析命令行参数，将结果存储在p变量中
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.CLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                print("k:", k)
                print("key:", key)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    # 随机种子
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()

