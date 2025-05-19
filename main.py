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

# 导入自定义库
from torchlight import DictAction  # 自定义的 argparse Action，用于将 key=value 形式的参数解析为字典

# 导入并设置系统资源限制（增加打开文件的最大数量）
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)  # 获取当前进程能打开的最大文件数限制
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))  # 设置新的最大文件数限制为 2048（或原上限，取较小者）


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
        description='Spatial Temporal Graph Convolution Network')  # 创建解析器对象，并设置描述信息

    # 工作目录参数
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',  # 默认工作目录
        help='the work folder for storing results')  # 参数说明

    # 保存模型的名称前缀（通常在运行时动态生成）
    parser.add_argument('-model_saved_name', default='')

    # 配置文件路径参数
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',  # 默认配置文件路径
        help='path to the configuration file')  # 参数说明

    # processor
    # === 处理器相关参数 ===
    parser.add_argument(
        '--phase', default='train', help='must be train or test')  # 运行阶段（训练或测试）
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

    # feeder
    # === 数据加载器 (Feeder) 相关参数 ===
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')  # 指定要使用的数据加载器类（字符串形式）
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,  # 默认使用 32 个工作进程加载数据
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
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')  # 用于网络初始化的预训练权重文件路径
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')  # 初始化时要忽略的权重名称（部分匹配）

    # optim
    # === 优化器 (Optimizer) 与训练策略相关参数 ===
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')  # 初始学习率
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],  # 默认在第 20, 40, 60 轮降低学习率
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')  # 降低学习率的轮次
    parser.add_argument(
        '--device',
        type=int,
        default=0,  # 默认使用 GPU 0
        nargs='+',
        help='the indexes of GPUs for training or testing')  # 用于训练或测试的 GPU 索引
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')  # 优化器类型（如 SGD, Adam）
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')  # 是否使用 Nesterov 动量（仅 SGD）
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')  # 训练批次大小
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')  # 测试批次大小
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
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')  # 优化器的权重衰减（L2 正则化）
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')  # 学习率衰减率
    parser.add_argument('--warm_up_epoch', type=int, default=0,
                        help='the epoch number for learning rate warm up')  # 学习率预热的轮数


    return parser


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
         用于基于骨骼的动作识别的处理类
    """

    def __init__(self, arg):
        self.arg = arg  # 保存传入的参数对象
        self.save_arg()  # 调用方法保存参数到文件

        # 根据运行阶段设置 TensorBoard writer
        if arg.phase == 'train':
            # 如果不是调试模式，则创建 train 和 val 的 writer
            if not arg.train_feeder_args.get('debug', False):  # 检查训练 feeder 参数中是否有 debug 标志
                # 设定 TensorBoard 日志保存路径
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                # 检查日志目录是否已存在
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')  # 询问用户是否删除
                    if answer.lower() == 'y':
                        shutil.rmtree(arg.model_saved_name)  # 删除已存在的目录
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')  # 提示用户刷新 TensorBoard
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                # 创建 TensorBoard writer
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                # 如果是调试模式，则 train 和 val writer 指向同一个测试目录
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.work_dir, 'test'), 'test')

        self.global_step = 0  # 初始化全局步数（用于 TensorBoard 记录）
        # pdb.set_trace() # 可以在此设置断点进行调试

        self.load_model()  # 加载模型

        # 如果不是仅计算模型大小的特殊阶段
        if self.arg.phase != 'model_size':
            self.load_optimizer()  # 加载优化器
            self.load_data()  # 加载数据  调用函数

        self.lr = self.arg.base_lr  # 初始化当前学习率
        self.best_acc = 0  # 初始化最佳准确率
        self.best_acc_epoch = 0  # 初始化最佳准确率对应的轮次

        # 将模型移动到指定的输出设备（通常是主 GPU）
        self.model = self.model.cuda(self.output_device)

        # 如果指定了多个 GPU，则使用 DataParallel 进行数据并行
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,  # 指定使用的 GPU ID 列表
                    output_device=self.output_device)  # 指定结果聚合到的设备

    def load_data(self):
        # 动态导入数据加载器类，从self.arg.feeder字符串指定的路径导入类
        # 例如：'feeders.feeder.Feeder'会导入feeders模块中的feeder.py文件中的Feeder类
        Feeder = import_class(self.arg.feeder)

        # 初始化一个空字典用于存储训练和测试的数据加载器
        self.data_loader = dict()

        # 如果当前是训练阶段，则需要创建用于训练的数据加载器
        if self.arg.phase == 'train':
            # 创建训练数据加载器并存储在字典中，键为'train'
            self.data_loader['train'] = torch.utils.data.DataLoader(
                # 使用导入的Feeder类创建数据集实例，将训练专用参数传递给它
                dataset=Feeder(**self.arg.train_feeder_args),
                # 设置每个批次的样本数量
                batch_size=self.arg.batch_size,
                # 启用数据随机打乱，提高训练的随机性
                shuffle=True,
                # 设置用于加载数据的子进程数量
                num_workers=self.arg.num_worker,
                # 丢弃不足一个批次的剩余数据
                drop_last=True,
                # 为每个worker设置随机种子的函数，确保数据加载的可重复性
                worker_init_fn=init_seed)

        # 无论是训练还是测试阶段，都需要创建测试数据加载器
        self.data_loader['test'] = torch.utils.data.DataLoader(
            # 使用导入的Feeder类创建数据集实例，将测试专用参数传递给它
            dataset=Feeder(**self.arg.test_feeder_args),
            # 设置测试时每个批次的样本数量（可能与训练时不同）
            batch_size=self.arg.test_batch_size,
            # 测试时不需要随机打乱数据
            shuffle=False,
            # 设置用于加载数据的子进程数量
            num_workers=self.arg.num_worker,
            # 测试时不丢弃剩余数据，以评估全部样本
            drop_last=False,
            # 为每个worker设置随机种子的函数，确保数据加载的可重复性
            worker_init_fn=init_seed)

    def load_model(self):
        # 确定输出设备（用于加载权重和计算损失）
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)  # 动态导入模型类
        # 将模型定义文件复制到工作目录，方便追溯
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)  # 打印模型类信息
        # 实例化模型，传入参数
        self.model = Model(**self.arg.model_args)
        print(self.model)  # 打印模型结构
        # 定义损失函数（交叉熵损失），并移动到输出设备
        self.loss = nn.CrossEntropyLoss().cuda(self.output_device)

        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
            # 将权重转换为 OrderedDict，移除 DataParallel 可能添加的 'module.' 前缀，并移动到输出设备
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            # 移除指定要忽略的权重
            for w in self.arg.ignore_weights:
                for key in list(weights.keys()):  # 迭代副本以允许在循环中删除
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Successfully Remove Weights: {}.'.format(key))
                        else:
                            # 这通常不会发生，因为 key 是从 weights.keys() 中获取的
                            self.print_log('Could Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()    # 获取当前模型的 state_dict
                diff = list(set(state.keys()).difference(set(weights.keys()))) # 找出模型有但权重文件没有的键
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                    # 尝试更新模型 state_dict（只加载匹配的键）
                    state.update(weights)  # 用加载的权重更新模型状态，这会覆盖匹配的键
                    self.model.load_state_dict(state)  # 使用 strict=False 允许部分加载

    # 加载优化器的方法
    def load_optimizer(self):
        # 根据参数选择优化器类型
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),  # 优化模型的所有参数
                lr=self.arg.base_lr,  # 设置初始学习率
                momentum=0.9,  # 设置动量
                nesterov=self.arg.nesterov,  # 是否使用 Nesterov 动量
                weight_decay=self.arg.weight_decay)  # 设置权重衰减
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer type')  # 如果优化器类型不支持，则报错

        # 打印学习率预热信息
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

        # 保存参数配置到文件的方法

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

    def adjust_learning_rate(self, epoch):
        # 支持 SGD 和 Adam 的学习率调整
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            # 处理学习率预热阶段
            if epoch < self.arg.warm_up_epoch:
                # 线性增加学习率
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

        # 打印当前时间的方法
    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    # 打印日志的方法
    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

        # 训练一个 epoch 的方法

    def train(self, epoch, save_model=False):
        self.model.train()  # 设置模型为训练模式（启用 dropout, batchnorm 更新等）
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']  # 获取训练数据加载器
        self.adjust_learning_rate(epoch)  # 调整当前 epoch 的学习率

        loss_value = []  # 用于记录每个 batch 的损失
        acc_value = []  # 用于记录每个 batch 的准确率
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()  # 开始计时
        # 初始化计时器字典，记录数据加载、模型计算和统计的时间
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)  # 初始化为小值避免除零
        # 使用 tqdm 创建进度条
        process = tqdm(loader, ncols=40)  # ncols 控制进度条宽度

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data)
            loss = self.loss(output, label)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        # 如果是训练阶段
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))  # 打印参数配置
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size  # 初始化全局步数
            # 定义一个函数计算模型的可训练参数数量
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            # 开始训练循环
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                # 执行一轮训练
                self.train(epoch, save_model=save_model)

                # 执行评估（每隔 eval_interval 轮 或 在最后一轮进行评估）
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model
            # 找到最佳准确率对应的模型权重文件
            # 注意：这里假设文件名包含轮次，并且只有一个匹配项
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            # 定义最终评估结果的文件名
            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')

            # 关闭文件日志记录，进行最终评估
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            # === 打印最终总结信息 ===
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        # 如果是测试阶段
        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            # 关闭文件日志记录进行测试评估
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))

            # 执行评估 (epoch 参数在这里意义不大，设为 0)
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    # 调用 get_parser 函数创建一个命令行参数解析器 parser。
    parser = get_parser()

    # load arg form config file
    # 解析命令行参数，将结果存储在p变量中
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f,Loader=yaml.CLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    # 随机种子
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
