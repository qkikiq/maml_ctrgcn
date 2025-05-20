#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction

# <<< META NETWORK INTEGRATION START >>>
# 可能需要 F 用于 MetaNetwork 中的某些操作，或者 Processor 中不需要
# import torch.nn.functional as F
# 导入我们创建的 MetaNetwork 类
from model.MetaNet import MetaNetwork
# <<< META NETWORK INTEGRATION END >>>

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True # 保持确定性，如果需要
    torch.backends.cudnn.benchmark = False   # 关闭 benchmark 以确保确定性

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

#解析参数
def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='') # 会在 init 中被覆盖

    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml', # 默认配置，需要修改为你自己的
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')


    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1, # 每多少个 epoch 保存一次
        help='the interval for storing models (#epoch)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=0, # 从哪个 epoch 开始保存，0 表示从第一个满足条件的epoch开始
        help='the start epoch to save model (#epoch)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5, # 每多少个 epoch 评估一次
        help='the interval for evaluating models (#epoch)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=4, # 减少默认 worker 数量，根据实际情况调整
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model (CTR-GCN related args are expected in the config file's model_args)
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.1, help='initial learning rate') # CTR-GCN常用0.1
    parser.add_argument(
        '--step',
        type=int,
        default=[30, 40], # CTR-GCN常用衰减点
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=True, help='use nesterov or not for SGD') # CTR-GCN常用True
    parser.add_argument(
        '--batch-size', type=int, default=64, help='training batch size') # CTR-GCN常用64或32
    parser.add_argument(
        '--test-batch-size', type=int, default=64, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=50, # CTR-GCN常用50或65
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005, # CTR-GCN常用 0.0005 或 0.0001
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=5) # CTR-GCN常用 5


    # 添加 MetaNetwork 相关参数
    parser.add_argument('--use-meta-graph',
                        type=str2bool,
                        default=False,
                        help='Whether to use meta-learning to generate graph per sample')
    parser.add_argument('--meta-net',
                        default='model.meta_graph.MetaGraphGenerator',
                        help='The class name of the meta network for graph generation')
    parser.add_argument('--meta-net-args',
                        action=DictAction,
                        default=dict(),
                        help='The arguments for the meta network')

    return parser


class Processor():
    """
        Processor for Skeleton-based Action Recognition (with MetaNetwork Integration)
    """
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        # <<< META NETWORK INTEGRATION START >>>
        # 在日志记录器初始化之后定义模型保存名称
        if arg.phase == 'train':
            # 如果不是调试模式
            # MODIFIED: 定义模型和日志保存的目录结构
            log_base_dir = os.path.join(arg.work_dir, 'logs') # 主日志目录
            model_save_base_dir = os.path.join(arg.work_dir, 'models') # 主模型保存目录

            # 使用 model_saved_name (如果命令行指定) 或时间戳创建唯一运行目录
            run_name = arg.model_saved_name if arg.model_saved_name else time.strftime('%Y%m%d_%H%M%S')
            self.run_log_dir = os.path.join(log_base_dir, run_name)
            self.run_model_dir = os.path.join(model_save_base_dir, run_name)
            arg.model_saved_name = self.run_model_dir # 更新 arg 中的路径，用于保存模型

            if not arg.train_feeder_args.get('debug', False): # 检查 debug 参数
                if os.path.exists(self.run_log_dir):
                    print(f'Log directory {self.run_log_dir} already exists.')
                    # answer = input('Delete it? (y/n): ')
                    # if answer.lower() == 'y':
                    #     shutil.rmtree(self.run_log_dir)
                    #     print(f'Log directory removed: {self.run_log_dir}')
                    # else:
                    #     print('Log directory not removed.')
                    #     # exit() # 或者退出，或者允许继续写入
                else:
                    os.makedirs(self.run_log_dir)
                    print(f'Created log directory: {self.run_log_dir}')

                if not os.path.exists(self.run_model_dir):
                     os.makedirs(self.run_model_dir)
                     print(f'Created model directory: {self.run_model_dir}')

                self.train_writer = SummaryWriter(os.path.join(self.run_log_dir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(self.run_log_dir, 'val'), 'val')
            else: # Debug 模式
                 debug_dir = os.path.join(arg.work_dir, 'debug', run_name)
                 os.makedirs(debug_dir, exist_ok=True)
                 self.train_writer = self.val_writer = SummaryWriter(debug_dir, 'debug')
                 arg.model_saved_name = debug_dir # Debug 模式下模型也保存在 debug 目录

        self.global_step = 0

        # ADDED: 初始化 MetaNetwork 实例 (在 load_model 之前或之中)
        self.meta_network = None
        if self.arg.use_meta_network:
            self.print_log("Initializing Meta Network...")
            # 确保 num_classes 和 num_nodes 可用
            num_classes = arg.model_args.get('num_class', None)
            # CTR-GCN 通常使用 graph_args.num_node
            num_nodes = arg.model_args.get('graph_args', {}).get('num_node', None)
            # 备选：直接从 model_args 获取 num_point/num_node
            if num_nodes is None:
                num_nodes = arg.model_args.get('num_point', arg.model_args.get('num_node', None))

            if num_classes is None or num_nodes is None:
                raise ValueError("Cannot initialize MetaNetwork: num_class or num_node not found in model_args or model_args.graph_args.")

            self.print_log(f"MetaNetwork params: in_channels={arg.inmeta_channels}, out_channels={arg.outmeta_channels}, num_nodes={num_nodes}, num_classes={num_classes}")

            self.meta_network = MetaNetwork(
                inmeta_channels=arg.inmeta_channels, # 需要确保这个值正确匹配输入数据 C
                outmeta_channels=arg.outmeta_channels,
                num_nodes=num_nodes,
                num_classes=num_classes,
                graph_init_type=arg.meta_graph_init_type
            )
            self.print_log("Meta Network Initialized.")
        # <<< META NETWORK INTEGRATION END >>>
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        # 加载主模型 (CTR-GCN)
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            # <<< META NETWORK INTEGRATION START >>>
            # MODIFIED: load_optimizer 现在会包含 MetaNetwork 的参数
            self.load_optimizer()
            # <<< META NETWORK INTEGRATION END >>>
            self.load_data()

        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        # <<< META NETWORK INTEGRATION START >>>
        # MODIFIED: 将模型和 meta_network (如果使用) 移动到设备
        # 确定主输出设备

        self.model = self.model.cuda(self.output_device)
        if self.meta_network:
            self.meta_network = self.meta_network.cuda(self.output_device)
            # 确保初始化的 generated_adj 在正确的设备上
            self.meta_network.generated_adj = self.meta_network.generated_adj.cuda(self.output_device)

        # 处理多GPU
        if type(self.arg.device) is list and len(self.arg.device) > 1:
            self.print_log(f"Using DataParallel for main model across devices: {self.arg.device}")
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.arg.device,
                output_device=self.output_device
            )
            # ADDED: 对 MetaNetwork 的 GCN 应用 DataParallel
            if self.meta_network and hasattr(self.meta_network, 'gcn'):
                 # 检查 MetaGCN 是否有参数
                 if list(self.meta_network.gcn.parameters()):
                     self.print_log(f"Using DataParallel for MetaNetwork GCN across devices: {self.arg.device}")
                     self.meta_network.gcn = nn.DataParallel(
                         self.meta_network.gcn,
                         device_ids=self.arg.device,
                         output_device=self.output_device
                     )
                 else:
                     self.print_log("MetaNetwork GCN has no parameters, skipping DataParallel for it.")
        # <<< META NETWORK INTEGRATION END >>>

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        # <<< META NETWORK INTEGRATION START >>>
        # ADDED: 检查数据通道数是否匹配 (可选的检查)
        # data_channel = self.arg.train_feeder_args.get('channel', 3) # 假设Feeder参数中有channel
        # if self.arg.use_meta_network and data_channel != self.arg.meta_gcn_in_channels:
        #     self.print_log(f"Warning: Data channel ({data_channel}) may not match meta GCN in_channels ({self.arg.meta_gcn_in_channels}). Verify data C dimension.")
        # <<< META NETWORK INTEGRATION END >>>

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed # 保持随机种子一致性
            )
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed # 保持随机种子一致性
        )
        self.print_log("Data loaded.")
        # 可以在这里打印样本形状以供调试
        # try:
        #     sample_data, _, _ = next(iter(self.data_loader['train' if 'train' in self.data_loader else 'test']))
        #     self.print_log(f"Sample data shape from DataLoader: {sample_data.shape}") # N, C, T, V, M
        # except Exception as e:
        #     self.print_log(f"Could not get sample data shape: {e}")


    def load_model(self):
        # <<< META NETWORK INTEGRATION START >>>
        # MODIFIED: 将 output_device 的确定移到 __init__ 中，这里直接使用
        output_device = self.output_device
        # <<< META NETWORK INTEGRATION END >>>

        Model = import_class(self.arg.model) # 导入 CTR-GCN 模型类
        # 复制模型文件到工作目录 (保持不变)
        try: # 尝试复制模型文件
            shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        except Exception as e:
            self.print_log(f"Warning: Could not copy model file. Error: {e}")

        self.print_log(f"Loading model: {self.arg.model}")
        self.print_log(f"Model args: {self.arg.model_args}")

        # 实例化主模型 (CTR-GCN)
        self.model = Model(**self.arg.model_args)
        # print(self.model) # 可以取消注释以打印模型结构

        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.print_log(f"Loss function: CrossEntropyLoss")

        # 加载预训练权重 (保持不变，但需要注意权重文件格式是否匹配，特别是用了MetaNetwork之后保存的格式)
        if self.arg.weights:
             self.global_step = 0 # 重置 global_step，稍后根据文件名或 checkpoint 信息恢复
             self.print_log('Load weights from {}.'.format(self.arg.weights))
             try:
                 # 尝试从文件名推断 global_step
                 step_part = self.arg.weights.split('_step-')[-1].split('.')[0]
                 if step_part.isdigit():
                     self.global_step = int(step_part)
                     self.print_log(f"Inferred global_step from weights filename: {self.global_step}")
             except:
                 self.print_log("Could not infer global_step from filename.")

             # <<< META NETWORK INTEGRATION START >>>
             # MODIFIED: 修改权重加载以处理可能包含 MetaNetwork 权重的字典
             saved_data = torch.load(self.arg.weights, map_location='cpu') # 先加载到 CPU

             if isinstance(saved_data, dict) and 'model_state_dict' in saved_data:
                 weights = saved_data['model_state_dict']
                 self.print_log("Loading weights from 'model_state_dict' key.")

                 # 加载 MetaNetwork GCN 权重 (如果存在且需要)
                 if self.arg.use_meta_network and self.meta_network and 'meta_gcn_state_dict' in saved_data:
                     meta_gcn_weights = saved_data['meta_gcn_state_dict']
                     meta_gcn_weights_cuda = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in meta_gcn_weights.items()])
                     try:
                         # 尝试加载到 self.meta_network.gcn (处理 DataParallel)
                         target_meta_gcn = self.meta_network.gcn.module if isinstance(self.meta_network.gcn, nn.DataParallel) else self.meta_network.gcn
                         target_meta_gcn.load_state_dict(meta_gcn_weights_cuda)
                         self.print_log("Successfully loaded MetaNetwork GCN weights.")
                     except Exception as e:
                         self.print_log(f"Warning: Failed to load MetaNetwork GCN weights: {e}")

                 # 加载生成的图状态 (如果保存了)
                 # if self.arg.use_meta_network and self.meta_network and 'generated_adj' in saved_data:
                 #    self.meta_network.generated_adj = saved_data['generated_adj'].cuda(output_device)
                 #    self.print_log("Successfully loaded generated graph state.")

             else:
                 # 假设是旧格式，直接是 state_dict
                 self.print_log("Weights file seems to be an old format (direct state_dict).")
                 weights = saved_data

             # 处理主模型权重
             weights_cuda = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

             # 忽略指定的权重 (保持不变)
             keys = list(weights_cuda.keys())
             for w in self.arg.ignore_weights:
                 for key in keys:
                     if w in key:
                         if weights_cuda.pop(key, None) is not None:
                             self.print_log('Successfully Remove Weights: {}.'.format(key))
                         else:
                             self.print_log('Warning: Can Not Remove Weights: {}.'.format(key))

             # 尝试加载主模型权重 (处理 DataParallel 前缀)
             try:
                 target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                 target_model.load_state_dict(weights_cuda)
                 self.print_log("Main model weights loaded successfully.")
             except RuntimeError as e: # 处理键不匹配等问题
                 self.print_log(f"Warning: Error loading main model weights: {e}")
                 self.print_log("Attempting to load with strict=False...")
                 try:
                     target_model.load_state_dict(weights_cuda, strict=False)
                     self.print_log("Main model weights loaded with strict=False.")
                 except Exception as e2:
                     self.print_log(f"Error loading with strict=False either: {e2}")
             except Exception as e:
                 self.print_log(f"An unexpected error occurred loading main model weights: {e}")

             # <<< META NETWORK INTEGRATION END >>>
        else:
             self.print_log("No pretrained weights specified for the main model.")

        # <<< META NETWORK INTEGRATION START >>>
        # ADDED: 检查主模型 forward 方法是否接受 adj 参数
        if self.arg.use_meta_network:
            forward_params = inspect.signature(self.model.module.forward if isinstance(self.model, nn.DataParallel) else self.model.forward).parameters
            if 'adj' not in forward_params:
                self.print_log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.print_log("CRITICAL WARNING: Main model's forward method does NOT accept 'adj' parameter.")
                self.print_log("MetaNetwork's generated graph CANNOT be passed to the model.")
                self.print_log(f"Please modify the forward method of your model class: {self.arg.model}")
                self.print_log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # 强烈建议在此处停止，因为 MetaNetwork 无效
                raise ValueError("Model forward method incompatible with MetaNetwork.")
            else:
                self.print_log("Main model's forward method accepts 'adj' parameter. Compatibility check passed.")
        # <<< META NETWORK INTEGRATION END >>>

    def load_optimizer(self):
        # <<< META NETWORK INTEGRATION START >>>
        # MODIFIED: 收集需要优化的参数列表
        params_to_optimize = []
        if hasattr(self.model, 'parameters'):
            params_to_optimize.extend(list(self.model.parameters()))
            self.print_log(f"Collected {len(params_to_optimize)} parameter groups from main model.")
        else:
             self.print_log("Warning: Main model does not seem to have parameters attribute.")

        # ADDED: 如果使用 MetaNetwork 并且其 GCN 需要训练，则添加到优化列表
        if self.arg.use_meta_network and self.meta_network and hasattr(self.meta_network, 'gcn'):
            # 确保 GCN 有参数
            meta_gcn_params = list(self.meta_network.gcn.parameters())
            if meta_gcn_params:
                 self.print_log(f"Adding {len(meta_gcn_params)} parameter groups from MetaNetwork GCN to optimizer.")
                 params_to_optimize.extend(meta_gcn_params)
            else:
                 self.print_log("MetaNetwork GCN has no trainable parameters.")

        if not params_to_optimize:
             raise ValueError("No parameters found to optimize.")
        # <<< META NETWORK INTEGRATION END >>>

        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params_to_optimize, # MODIFIED: 使用收集到的参数列表
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
             # Adam 通常需要较低的学习率，例如 0.001 或更低
             adam_lr = self.arg.base_lr if self.arg.base_lr < 0.01 else 0.001
             if self.arg.base_lr >= 0.01:
                 self.print_log(f"Warning: High base_lr ({self.arg.base_lr}) for Adam. Using {adam_lr} instead.")
             self.optimizer = optim.Adam(
                 params_to_optimize, # MODIFIED: 使用收集到的参数列表
                 lr=adam_lr,
                 weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")

        self.print_log(f"Optimizer loaded: {self.arg.optimizer} with initial LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        if self.arg.warm_up_epoch > 0:
             self.print_log(f'Using learning rate warm up for {self.arg.warm_up_epoch} epochs.')

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        # <<< META NETWORK INTEGRATION START >>>
        # MODIFIED: 保存到当前运行的日志目录中
        config_path = os.path.join(self.run_log_dir if hasattr(self, 'run_log_dir') else self.arg.work_dir, 'config.yaml')
        # <<< META NETWORK INTEGRATION END >>>
        try:
            with open(config_path, 'w') as f:
                f.write(f"# command line: {' '.join(sys.argv)}\n\n")
                yaml.dump(arg_dict, f, default_flow_style=False) # 使用更易读的格式
            self.print_log(f"Arguments saved to {config_path}")
        except Exception as e:
            self.print_log(f"Warning: Failed to save arguments to {config_path}. Error: {e}")


    def adjust_learning_rate(self, epoch):
        # 基本逻辑保持不变，但确保 optimizer 是存在的
        if self.optimizer is None:
            self.print_log("Optimizer not initialized, cannot adjust learning rate.")
            return self.arg.base_lr # 返回基础学习率

        if self.arg.optimizer in ['SGD', 'Adam']: # Adam 也可以用同样的衰减策略
            if epoch < self.arg.warm_up_epoch:
                # 线性预热
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                # Step-wise 衰减
                decay_factor = self.arg.lr_decay_rate ** np.sum((epoch - self.arg.warm_up_epoch) >= np.array(self.arg.step))
                lr = self.arg.base_lr * decay_factor

            # 更新优化器中的学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            # 如果添加了其他优化器，需要在此处添加相应的学习率调整逻辑
            raise ValueError(f"Learning rate adjustment not implemented for optimizer: {self.arg.optimizer}")

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, msg, print_time=True):
        if print_time:
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            msg = f"[{localtime}] {msg}"
        print(msg)
        # <<< META NETWORK INTEGRATION START >>>
        # MODIFIED: 写入当前运行的日志文件
        log_file_path = os.path.join(self.run_log_dir if hasattr(self, 'run_log_dir') else self.arg.work_dir, 'log.txt')
        # <<< META NETWORK INTEGRATION END >>>
        if self.arg.print_log:
            try:
                with open(log_file_path, 'a') as f:
                    print(msg, file=f)
            except Exception as e:
                # 第一次打印时尝试创建目录
                if not os.path.exists(os.path.dirname(log_file_path)):
                     try:
                         os.makedirs(os.path.dirname(log_file_path))
                         with open(log_file_path, 'a') as f:
                             print(msg, file=f)
                     except Exception as e_inner:
                         print(f"Error: Could not create log directory or write to log file {log_file_path}. Inner Error: {e_inner}")
                else:
                    print(f"Error: Could not write to log file {log_file_path}. Error: {e}")


    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        # <<< META NETWORK INTEGRATION START >>>
        # MODIFIED: 设置模型和 MetaNetwork (如果使用) 为训练模式
        self.model.train()
        if self.arg.use_meta_network and self.meta_network:
            self.meta_network.train() # 确保 Meta GCN 也处于训练模式
        # <<< META NETWORK INTEGRATION END >>>

        self.print_log(f'Training epoch: {epoch + 1}')
        loader = self.data_loader['train']
        current_lr = self.adjust_learning_rate(epoch) # 获取调整后的学习率
        self.print_log(f'Epoch {epoch+1} Learning Rate: {current_lr:.8f}') # 提高精度

        loss_value = []
        acc_value = []
        # <<< META NETWORK INTEGRATION START >>>
        # ADDED: 用于临时存储本epoch的特征和标签以更新LDA
        if self.arg.use_meta_network:
            epoch_features = []
            epoch_labels = []
        # <<< META NETWORK INTEGRATION END >>>

        self.train_writer.add_scalar('epoch', epoch, self.global_step) # global_step 现在在 batch 循环中更新
        self.record_time()
        # <<< META NETWORK INTEGRATION START >>>
        # MODIFIED: 添加 meta_features 计时器
        timer = dict(dataloader=0.001, meta_features=0.001, model=0.001, statistics=0.001)
        # <<< META NETWORK INTEGRATION END >>>
        process = tqdm(loader, ncols=100, desc=f"Epoch {epoch+1} Train") # 调整宽度和描述

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # 计时开始 (数据加载)
            timer['dataloader'] += self.split_time()

            # 移动数据到设备
            data = data.float().cuda(self.output_device) # N, C, T, V, M
            label = label.long().cuda(self.output_device) # N

            # <<< META NETWORK INTEGRATION START >>>
            # --- Meta Network 处理 ---
            generated_adj = None
            if self.arg.use_meta_network and self.meta_network:
                # 1. 使用 Meta GCN 提取特征
                # 处理多人数据 (M维度) - 假设 MetaGCN 输入是 (N, C, T, V)
                if data.dim() == 5: # N, C, T, V, M
                    # 简单策略：取第一个人
                    meta_input_data = data[:, :, :, :, 0].contiguous()
                    # 你可以选择其他策略，如平均 M 维度或修改 MetaGCN
                elif data.dim() == 4: # N, C, T, V
                     meta_input_data = data
                else:
                     raise ValueError(f"Unexpected data dimension: {data.shape}")

                # 检查通道数
                if meta_input_data.shape[1] != self.arg.meta_gcn_in_channels:
                     raise ValueError(f"Data channel {meta_input_data.shape[1]} != meta GCN in_channels {self.arg.meta_gcn_in_channels}")

                with torch.no_grad(): # 如果MetaGCN不参与主模型梯度，可以no_grad加速
                      meta_features = self.meta_network.extract_features(meta_input_data) # N, meta_gcn_out_channels
                timer['meta_features'] += self.split_time() # 记录特征提取时间

                # 2. 收集特征和标签用于后续LDA更新 (移到 CPU 避免 GPU 内存积累)
                epoch_features.append(meta_features.detach().cpu().numpy())
                epoch_labels.append(label.detach().cpu().numpy())

                # 3. 获取当前生成的图 (确保在正确的设备上)
                generated_adj = self.meta_network.get_graph().to(self.output_device)

            # --- 主模型前向传播 ---
            # MODIFIED: 将 generated_adj (如果存在) 传入 CTR-GCN
            if generated_adj is not None:
                 output = self.model(data, adj=generated_adj) # N, num_classes
            else:
                 output = self.model(data) # N, num_classes
            # <<< META NETWORK INTEGRATION END >>>

            # 处理可能的辅助损失 (CTR-GCN 通常没有，但以防万一)
            if isinstance(output, tuple):
                 output, aux_loss = output
                 loss = self.loss(output, label) + aux_loss * 0.1 # 示例：加权辅助损失
            else:
                 loss = self.loss(output, label)

            # --- 反向传播和优化 ---
            self.optimizer.zero_grad()
            loss.backward()
            # 可以添加梯度裁剪 (可选，但有时有用)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            timer['model'] += self.split_time() # 记录模型前向+反向+优化时间

            # --- 统计与记录 ---
            loss_value.append(loss.item())
            with torch.no_grad(): # 计算准确率不需要梯度
                 value, predict_label = torch.max(output.data, 1)
                 acc = torch.mean((predict_label == label.data).float())
                 acc_value.append(acc.item())

            # 更新 TensorBoard (更频繁地更新可能有助于观察)
            if batch_idx % self.arg.log_interval == 0:
                 self.train_writer.add_scalar('batch/acc', acc.item(), self.global_step)
                 self.train_writer.add_scalar('batch/loss', loss.item(), self.global_step)
                 self.train_writer.add_scalar('batch/lr', current_lr, self.global_step)

            # 更新tqdm进度条描述
            process.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.2f}", lr=f"{current_lr:.6f}")

            timer['statistics'] += self.split_time()
            # ------- Batch End --------

        # --- Epoch End ---
        # 在 Epoch 结束时记录平均指标到 TensorBoard
        mean_loss = np.mean(loss_value)
        mean_acc = np.mean(acc_value)
        self.train_writer.add_scalar('epoch/mean_loss', mean_loss, epoch)
        self.train_writer.add_scalar('epoch/mean_acc', mean_acc, epoch)

        # 打印时间消耗比例
        total_time = sum(timer.values())
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / total_time)))
            for k, v in timer.items() if total_time > 0 # 避免除零
        }
        self.print_log(
            f'\tMean training loss: {mean_loss:.4f}. Mean training acc: {mean_acc*100:.2f}%.')
        time_log_str = ', '.join([f'[{k}]{v}' for k, v in proportion.items()])
        self.print_log(f'\tTime consumption ({total_time:.2f}s): {time_log_str}')

        # <<< META NETWORK INTEGRATION START >>>
        # --- 更新 MetaNetwork 的 LDA 和 Graph ---
        if self.arg.use_meta_network and self.meta_network and \
           (epoch + 1) % self.arg.meta_lda_update_epoch == 0 and \
           epoch_features and epoch_labels: # 确保收集到了数据

            self.print_log(f"Epoch {epoch+1}: Triggering MetaNetwork LDA and graph update...")
            # 将本epoch收集的数据添加到buffer (合并数据)
            try:
                features_to_add = np.concatenate(epoch_features, axis=0)
                labels_to_add = np.concatenate(epoch_labels, axis=0)
                self.meta_network.add_to_buffer(features_to_add, labels_to_add)
                self.print_log(f"Added {features_to_add.shape[0]} samples to MetaNetwork buffer.")

                # 执行更新
                updated = self.meta_network.update_lda_and_graph(self.output_device)
                if updated:
                    self.print_log("MetaNetwork LDA and graph updated successfully.")
                    # 可选：记录图的统计信息
                    current_adj = self.meta_network.get_graph()
                    self.train_writer.add_scalar('meta/adj_mean', current_adj.mean(), self.global_step)
                    self.train_writer.add_scalar('meta/adj_std', current_adj.std(), self.global_step)
                    # self.train_writer.add_histogram('meta/adj_dist', current_adj, self.global_step) # 可能较慢
                else:
                    self.print_log("MetaNetwork update skipped or failed based on internal logic.")
            except Exception as e:
                 self.print_log(f"Error during MetaNetwork update process: {e}")
                 traceback.print_exc()
        # <<< META NETWORK INTEGRATION END >>>

        # --- 保存模型 ---
        if save_model:
            # <<< META NETWORK INTEGRATION START >>>
            # MODIFIED: 保存包含主模型和 Meta GCN (如果需要) 的字典
            model_state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
            model_weights = OrderedDict([[k, v.cpu()] for k, v in model_state_dict.items()]) # 保存到 CPU

            save_dict = {'model_state_dict': model_weights}
            self.print_log(f"Prepared main model state dict for saving.")

            # ADDED: 如果 MetaNetwork 的 GCN 是可训练的，并且希望保存其权重
            if self.arg.use_meta_network and self.meta_network and hasattr(self.meta_network, 'gcn'):
                 meta_gcn_target = self.meta_network.gcn.module if isinstance(self.meta_network.gcn, nn.DataParallel) else self.meta_network.gcn
                 # 检查是否有参数
                 if list(meta_gcn_target.parameters()):
                     meta_gcn_state_dict = meta_gcn_target.state_dict()
                     meta_gcn_weights = OrderedDict([[f"{k}", v.cpu()] for k, v in meta_gcn_state_dict.items()]) # Key 不加前缀
                     save_dict['meta_gcn_state_dict'] = meta_gcn_weights
                     self.print_log(f"Added MetaNetwork GCN state dict to save_dict.")
                 else:
                     self.print_log("Skipping saving MetaNetwork GCN state (no parameters).")

            # ADDED: 可选，保存生成的邻接矩阵的状态
            if self.arg.use_meta_network and self.meta_network:
               save_dict['generated_adj'] = self.meta_network.get_graph().cpu()
               self.print_log(f"Added generated graph state to save_dict.")

            # 文件名包含 epoch 和 global_step
            # MODIFIED: 保存到特定运行的模型目录
            save_path = os.path.join(self.run_model_dir, f'epoch-{epoch+1}_step-{int(self.global_step)}.pt')
            torch.save(save_dict, save_path)
            self.print_log(f'Model saved to: {save_path}')
            # <<< META NETWORK INTEGRATION END >>>

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # 文件操作保持不变
        if wrong_file is not None:
            wf_path = os.path.join(self.run_log_dir if hasattr(self, 'run_log_dir') else self.arg.work_dir, wrong_file)
            f_w = open(wf_path, 'w')
            self.print_log(f"Writing wrong predictions to: {wf_path}")
        else:
            f_w = None
        if result_file is not None:
            rf_path = os.path.join(self.run_log_dir if hasattr(self, 'run_log_dir') else self.arg.work_dir, result_file)
            f_r = open(rf_path, 'w')
            self.print_log(f"Writing all predictions to: {rf_path}")
        else:
            f_r = None

        # <<< META NETWORK INTEGRATION START >>>
        # MODIFIED: 设置模型和 MetaNetwork (如果使用) 为评估模式
        self.model.eval()
        if self.arg.use_meta_network and self.meta_network:
            self.meta_network.eval() # 确保 Meta GCN 处于评估模式
        # <<< META NETWORK INTEGRATION END >>>

        self.print_log(f'Evaluation epoch: {epoch + 1}')

        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=100, desc=f"Epoch {epoch+1} Eval {ln}")

            # <<< META NETWORK INTEGRATION START >>>
            # ADDED: 在评估时，获取固定的生成图 (通常是训练结束时的状态或加载的状态)
            generated_adj = None
            if self.arg.use_meta_network and self.meta_network:
                generated_adj = self.meta_network.get_graph().to(self.output_device)
                self.print_log(f"Using fixed generated graph (shape: {generated_adj.shape}) for evaluation.")
            # <<< META NETWORK INTEGRATION END >>>

            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label.numpy()) # 直接用 numpy 存储标签

                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    # <<< META NETWORK INTEGRATION START >>>
                    # --- 模型前向传播 ---
                    # MODIFIED: 将 generated_adj (如果存在) 传入 CTR-GCN
                    if generated_adj is not None:
                        output = self.model(data, adj=generated_adj)
                    else:
                        output = self.model(data)
                    # <<< META NETWORK INTEGRATION END >>>

                    # 处理可能的辅助损失 (评估时通常只关心主输出)
                    if isinstance(output, tuple):
                        output = output[0]

                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy()) # 移动到 CPU 避免累积
                    loss_value.append(loss.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy()) # 移动到 CPU
                    step += 1

                # --- 保存错误/正确预测 ---
                if f_w is not None or f_r is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        # 保存所有结果
                        if f_r is not None:
                            # 写入样本索引（如果可用）和预测/真实标签
                            sample_id = index[i] if isinstance(index, (list, tuple, np.ndarray)) else batch_idx * self.arg.test_batch_size + i
                            f_r.write(f"{sample_id},{x},{true[i]}\n")
                        # 保存错误结果
                        if x != true[i] and f_w is not None:
                            sample_id = index[i] if isinstance(index, (list, tuple, np.ndarray)) else batch_idx * self.arg.test_batch_size + i
                            f_w.write(f"{sample_id},{x},{true[i]}\n")

            # --- 评估结果处理 ---
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            # 获取样本名称用于保存分数（如果 Feeder 提供）
            sample_names = self.data_loader[ln].dataset.sample_name if hasattr(self.data_loader[ln].dataset, 'sample_name') else np.arange(len(score))

            accuracy = self.data_loader[ln].dataset.top_k(score, 1) # Top-1

            if self.arg.phase == 'train': # 只在训练阶段更新最佳准确率和 TensorBoard
                if accuracy > self.best_acc:
                    self.best_acc = accuracy
                    self.best_acc_epoch = epoch + 1
                    self.print_log(f"**** New Best Top-1 Accuracy on {ln}: {self.best_acc*100:.2f}% at epoch {self.best_acc_epoch} ****")
                    # <<< META NETWORK INTEGRATION START >>>
                    # ADDED: 保存最佳模型 (包含 Meta GCN 和图状态)
                    best_model_path = os.path.join(self.run_model_dir, 'best_model.pt')
                    model_state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                    model_weights = OrderedDict([[k, v.cpu()] for k, v in model_state_dict.items()])
                    save_dict = {'model_state_dict': model_weights, 'epoch': epoch + 1, 'accuracy': accuracy}

                    if self.arg.use_meta_network and self.meta_network:
                         meta_gcn_target = self.meta_network.gcn.module if isinstance(self.meta_network.gcn, nn.DataParallel) else self.meta_network.gcn
                         if list(meta_gcn_target.parameters()):
                             meta_gcn_state_dict = meta_gcn_target.state_dict()
                             meta_gcn_weights = OrderedDict([[f"{k}", v.cpu()] for k, v in meta_gcn_state_dict.items()])
                             save_dict['meta_gcn_state_dict'] = meta_gcn_weights
                         save_dict['generated_adj'] = self.meta_network.get_graph().cpu()

                    torch.save(save_dict, best_model_path)
                    self.print_log(f"Saved new best model checkpoint to {best_model_path}")
                    # <<< META NETWORK INTEGRATION END >>>


                # 记录验证集指标到 TensorBoard (使用 global_step 对齐训练曲线)
                self.val_writer.add_scalar(f'{ln}/loss', loss, self.global_step)
                self.val_writer.add_scalar(f'{ln}/acc_top1', accuracy, self.global_step)
                for k in self.arg.show_topk:
                     if k != 1: # Top-1 已经记录
                         topk_acc = self.data_loader[ln].dataset.top_k(score, k)
                         self.val_writer.add_scalar(f'{ln}/acc_top{k}', topk_acc, self.global_step)


            self.print_log(f'\tMean {ln} loss of {step} batches: {loss:.4f}.')
            # 确保 sample_names 和 score 长度一致
            if len(sample_names) != len(score):
                self.print_log(f"Warning: Length mismatch between sample names ({len(sample_names)}) and scores ({len(score)}). Using indices for score dict.")
                sample_names = np.arange(len(score))
            score_dict = dict(zip(sample_names, score))

            for k in self.arg.show_topk:
                topk_acc = self.data_loader[ln].dataset.top_k(score, k)
                self.print_log(f'\tTop-{k} accuracy on {ln}: {topk_acc*100:.2f}%')

            # 保存分数 (保持不变)
            if save_score:
                score_path = os.path.join(self.run_log_dir if hasattr(self, 'run_log_dir') else self.arg.work_dir, f'epoch{epoch+1}_{ln}_score.pkl')
                with open(score_path, 'wb') as f:
                    pickle.dump(score_dict, f)
                self.print_log(f"Scores saved to {score_path}")

            # 计算混淆矩阵和每个类的准确率 (保持不变)
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            try:
                confusion = confusion_matrix(label_list, pred_list)
                list_diag = np.diag(confusion)
                list_raw_sum = np.sum(confusion, axis=1)
                each_acc = np.nan_to_num(list_diag / list_raw_sum) # 处理除零 (如果某类从未出现)
                self.print_log(f'\tAccuracy for each class on {ln}:')
                # 打印前几个和后几个类别的准确率
                acc_str = ", ".join([f"{acc*100:.2f}%" for acc in each_acc[:5]]) + " ... " + ", ".join([f"{acc*100:.2f}%" for acc in each_acc[-5:]])
                self.print_log(f'\t  [{acc_str}]')

                # 保存每个类的准确率和混淆矩阵
                csv_path = os.path.join(self.run_log_dir if hasattr(self, 'run_log_dir') else self.arg.work_dir, f'epoch{epoch+1}_{ln}_acc_conf.csv')
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Class Accuracy'] + [f'Class_{i}' for i in range(len(each_acc))])
                    writer.writerow([''] + list(each_acc))
                    writer.writerow([]) # 空行分隔
                    writer.writerow(['Confusion Matrix (True\\Pred)'] + [f'Pred_{i}' for i in range(confusion.shape[1])])
                    for i, row in enumerate(confusion):
                        writer.writerow([f'True_{i}'] + list(row))
                self.print_log(f"Class accuracies and confusion matrix saved to {csv_path}")

            except Exception as e:
                self.print_log(f"Error calculating/saving confusion matrix: {e}")

        # 关闭文件
        if f_w is not None: f_w.close()
        if f_r is not None: f_r.close()


    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))  # 打印参数配置
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size  # 初始化全局步数


            # <<< META NETWORK INTEGRATION START >>>
            # MODIFIED: 计算并打印总参数量
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            # 初始化total_params并设置为main_params的值
            total_params = count_parameters(self.model)

            meta_params_count = 0
            if self.arg.use_meta_network and self.meta_network:
                 meta_gcn_params = count_parameters(self.meta_network.gcn)
                 self.print_log(f"# Meta Network GCN Trainable Parameters: {meta_gcn_params:,}")
                 total_params += meta_gcn_params
                 self.print_log(f"# Total Trainable Parameters: {total_params:,}")
            # <<< META NETWORK INTEGRATION END >>>

            # 开始训练循环
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # MODIFIED: 调整保存逻辑，>= save_epoch
                save_model_this_epoch = (((epoch + 1) % self.arg.save_interval == 0) or \
                                          (epoch + 1 == self.arg.num_epoch)) and \
                                         (epoch + 1) >= self.arg.save_epoch

                self.train(epoch, save_model=save_model_this_epoch)

                # 控制评估频率
                if (epoch + 1) % self.arg.eval_interval == 0 or \
                   epoch + 1 == self.arg.num_epoch:
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test']) # 默认在 test 上评估

            # --- 训练结束后的最佳模型测试 ---
            self.print_log(f"Training finished. Best Top-1 accuracy on test set: {self.best_acc*100:.2f}% at epoch {self.best_acc_epoch}")
            best_model_path = os.path.join(self.run_model_dir, 'best_model.pt') # 指向保存的最佳模型
            if self.best_acc_epoch > 0 and os.path.exists(best_model_path):
                 self.print_log(f"Loading best model weights from: {best_model_path} for final evaluation...")
                 try:
                     # <<< META NETWORK INTEGRATION START >>>
                     # MODIFIED: 加载包含 MetaNetwork 状态的最佳模型字典
                     saved_data = torch.load(best_model_path, map_location='cpu') # 加载到 CPU

                     # 加载主模型权重
                     model_weights = saved_data.get('model_state_dict')
                     if model_weights:
                         # 处理可能的 DataParallel 前缀 (如果模型实例是 DP 包装的)
                         target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                         target_model.load_state_dict(model_weights)
                         self.print_log("Loaded best main model weights.")
                     else:
                         raise ValueError("Best model file does not contain 'model_state_dict'.")

                     # 加载 MetaNetwork GCN 权重 (如果保存了且使用了)
                     if self.arg.use_meta_network and self.meta_network and 'meta_gcn_state_dict' in saved_data:
                         meta_gcn_weights = saved_data['meta_gcn_state_dict']
                         target_meta_gcn = self.meta_network.gcn.module if isinstance(self.meta_network.gcn, nn.DataParallel) else self.meta_network.gcn
                         if list(target_meta_gcn.parameters()): # 检查是否有参数
                             target_meta_gcn.load_state_dict(meta_gcn_weights)
                             self.print_log("Loaded best MetaNetwork GCN weights.")
                         else:
                              self.print_log("Skipped loading best MetaNetwork GCN weights (no parameters).")


                     # 加载生成的图状态 (如果保存了)
                     if self.arg.use_meta_network and self.meta_network and 'generated_adj' in saved_data:
                        self.meta_network.generated_adj = saved_data['generated_adj'].cuda(self.output_device)
                        self.print_log("Loaded best generated graph state.")
                     # <<< META NETWORK INTEGRATION END >>>

                     # 设置文件路径并执行最终评估
                     wf = 'best_model_wrong.txt' # 保存到日志目录
                     rf = 'best_model_right.txt' # 保存到日志目录
                     self.arg.print_log = True # 确保日志开启
                     self.print_log("Running final evaluation with best model weights...")
                     # 使用 best_acc_epoch-1 是因为 eval 的 epoch 参数是 0-based
                     self.eval(epoch=self.best_acc_epoch - 1, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)

                 except Exception as e:
                     self.print_log(f"Error during final evaluation with best model: {e}")
                     traceback.print_exc()
            else:
                 self.print_log("Best model checkpoint not found or best epoch is 0. Skipping final evaluation with best model.")

            # 打印最终的总结信息
            self.print_log(f'Final Best Top-1 Accuracy: {self.best_acc * 100:.2f}%')
            self.print_log(f'Best Epoch: {self.best_acc_epoch}')
            self.print_log(f'Log Directory: {self.run_log_dir}')
            self.print_log(f'Model Directory: {self.run_model_dir}')
            # 其他你想记录的信息...

        elif self.arg.phase == 'test':
            self.print_log('Starting testing phase...')
            if self.arg.weights is None:
                raise ValueError('Please specify --weights for testing.')
            if not os.path.exists(self.arg.weights):
                 raise FileNotFoundError(f"Weights file not found: {self.arg.weights}")


            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')

            # <<< META NETWORK INTEGRATION START >>>
            # MODIFIED: 加载包含 MetaNetwork 状态的权重文件
            try:
                 saved_data = torch.load(self.arg.weights, map_location='cpu') # 加载到 CPU

                 # 加载主模型权重
                 model_weights = saved_data.get('model_state_dict')
                 if model_weights:
                     target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                     target_model.load_state_dict(model_weights)
                     self.print_log("Loaded main model weights for testing.")
                 else:
                     # 尝试旧格式
                     try:
                         self.print_log("Weights file might be old format, trying direct load...")
                         target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                         target_model.load_state_dict(saved_data)
                         self.print_log("Loaded weights assuming old format (direct state dict).")
                     except:
                         raise ValueError("Could not load model state_dict from the weights file.")

                 # 加载 MetaNetwork GCN 权重 (如果使用了且存在)
                 if self.arg.use_meta_network and self.meta_network:
                     if 'meta_gcn_state_dict' in saved_data:
                         meta_gcn_weights = saved_data['meta_gcn_state_dict']
                         target_meta_gcn = self.meta_network.gcn.module if isinstance(self.meta_network.gcn, nn.DataParallel) else self.meta_network.gcn
                         if list(target_meta_gcn.parameters()):
                             target_meta_gcn.load_state_dict(meta_gcn_weights)
                             self.print_log("Loaded MetaNetwork GCN weights for testing.")
                         else:
                             self.print_log("Skipped loading MetaNetwork GCN weights (no parameters).")

                 # 加载生成的图状态 (如果使用了且存在)
                 if self.arg.use_meta_network and self.meta_network:
                     if 'generated_adj' in saved_data:
                         self.meta_network.generated_adj = saved_data['generated_adj'].cuda(self.output_device)
                         self.print_log("Loaded generated graph state for testing.")
                     else:
                         # 如果 checkpoint 没有图状态，使用 MetaNetwork 初始化时的图
                         self.print_log("Generated graph state not found in checkpoint, using the initial graph from MetaNetwork.")

            except Exception as e:
                 self.print_log(f"Error loading weights file {self.arg.weights}: {e}")
                 traceback.print_exc()
                 raise e
            # <<< META NETWORK INTEGRATION END >>>

            # 定义测试结果文件名
            wf = os.path.basename(self.arg.weights).replace('.pt', '_test_wrong.txt')
            rf = os.path.basename(self.arg.weights).replace('.pt', '_test_right.txt')

            self.arg.print_log = True # 确保测试日志开启
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Testing Done.\n')

        elif self.arg.phase == 'model_size':
            # <<< META NETWORK INTEGRATION START >>>
            # ADDED: 处理 model_size 阶段
            self.print_log("Calculating model size...")
            def count_parameters(model):
                 if model is None: return 0
                 return sum(p.numel() for p in model.parameters() if p.requires_grad)
            main_params = count_parameters(self.model)
            self.print_log(f"# Main Model Trainable Parameters: {main_params:,}")
            total_params = main_params
            if self.arg.use_meta_network and self.meta_network:
                 meta_params = count_parameters(self.meta_network.gcn)
                 self.print_log(f"# Meta Network GCN Trainable Parameters: {meta_params:,}")
                 total_params += meta_params
                 self.print_log(f"# Total Trainable Parameters: {total_params:,}")
            # <<< META NETWORK INTEGRATION END >>>
        else:
             raise ValueError(f"Unsupported phase: {self.arg.phase}. Must be 'train', 'test', or 'model_size'.")


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
             default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args() # 重新解析，应用配置文件默认值，并允许命令行覆盖
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()