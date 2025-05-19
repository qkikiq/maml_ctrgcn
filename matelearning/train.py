import argparse
import os

import torch

from utils import set_gpu, check_dir, log, Timer


def get_dataset(options):
    if options.dataset == 'ntu60':
        from mate_data.ntu60 import ntu60, NTU60train, FewShotDataloader

        # 创建元训练数据集 + 将最初的训练集拎出来测试集作为元数据集整体
        dataset_train = NTU60train(
            ntu60_dataset=ntu60(data_path='../data/ntu/NTU60_CS.npz',
                                split='train'),
            phase='train'

        )
        # dataset_meta = ntu60(data_path='../data/ntu/NTU60_CS.npz',
        #                       split='train')
        # 创建元测试数据集
        dataset_test = NTU60train(
            ntu60_dataset=ntu60(data_path='../data/ntu/NTU60_CS.npz',
                                split='train'),
            phase='test'
        )
        data_loader = FewShotDataloader
    # elif options.dataset == 'tieredImageNet':
    #     from data.tiered_imagenet import tieredImageNet, FewShotDataloader
    #     dataset_train = tieredImageNet(phase='train')
    #     dataset_val = tieredImageNet(phase='val')
    #     data_loader = FewShotDataloader

    else:
        print("Cannot recognize the dataset type")
        assert (False)

    # return (dataset_meta, dataset_test, data_loader)
    return (dataset_train, dataset_test, data_loader)


if __name__ == '__main__':
    # 网络训练参数
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60,  # 训练总论数
                        help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,  # 保存模型的间隔
                        help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,  # 每个训练类的支持样本数
                        help='number of support examples per training class')
    parser.add_argument('--test-shot', type=int, default=5,  # 每个验证类的支持样本数
                        help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,  # 每个训练类的查询样本数
                        help='number of query examples per training class')
    parser.add_argument('--test-episode', type=int, default=2000,  # 验证时的episode数
                        help='number of episodes per validation')
    parser.add_argument('--test-query', type=int, default=15,  # 每个验证类的查询样本数
                        help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,  # 每个训练类的类别数
                        help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,  # 每个验证类的类别数
                        help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')  # 保存模型的路径
    parser.add_argument('--load', default=None, help='path of the checkpoint file')  # 加载模型的路径
    parser.add_argument('--gpu', default='0, 1, 2, 3')  # 使用的GPU
    parser.add_argument('--network', type=str, default='ProtoNet',  # 选择的嵌入网络   ？？？？？
                        help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',  # 选择的分类头   ？？？？？
                        help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='ntu60',  # 选择的数据集   ？？？？？
                        help='choose which classification head to use. ntu60, ntu120, ')
    parser.add_argument('--episodes-per-batch', type=int, default=8,  # 每个 batch 中包含的 episode 数量
                        help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,  # 标签平滑（Label Smoothing）中的ε参数，
                        help='epsilon of label smoothing')
    parser.add_argument('--task-embedding', type=str, default='None',  # 选择的任务嵌入类型   ？？？？？
                        help='choose which type of task embedding will be used')
    parser.add_argument('--start-epoch', type=int, default=-1,  # 指定从第几轮开始使用 task embedding，默认-1（不使用）
                        help='choose when to use task embedding')
    parser.add_argument('--post-processing', type=str, default='None',  # 指定是否对样本嵌入进行后处理（比如加一个后处理网络），默认不使用
                        help='use an extra post processing net for sample embeddings')

    opt = parser.parse_args()  # 解析参数

    (dataset_train, dataset_test, data_loader) = get_dataset(opt)  # 加载数据集

    # dloader_train = data_loader(
    #     dataset=dataset_meta,  # 元数据集
    #     nKnovel=opt.train_way,  # 每个episode中类别数
    #     nKbase=0,  # 每个episode中基类数  不从base类中选任务
    #     nSupports=opt.train_shot,  # num training examples per novel category   每个类别中支持的样本数（shot）
    #     nTestNovel=opt.train_way * opt.train_query,
    #     # num test examples for all the novel categories     所有新类别上的 query 样本数；通常设为 way × query
    #     nTestBase=0,  # num test examples for all the base categories
    #     batch_size=opt.episodes_per_batch,  # 每个 batch 中有多少个 episodes
    #     num_workers=4,
    #     epoch_size=opt.episodes_per_batch * 1000,  # num of batches per epoch  每个 epoch 生成多少个 episodes
    # )

    dloader_train = data_loader(
        dataset=dataset_train,  # 元数据集实例
        nKnovel=opt.train_way,  # 每个episode包含的新类别数量 (K-way)
        nKbase=0,  # 不使用基类
        nSupport=opt.train_shot,  # 每个新类别的支持集样本数 (N-shot)
        nQueryNovel=opt.train_way * opt.train_query,  # 新类别的查询集总样本数 = 类别数 × 每类查询样本数
        nQueryBase=0,  # 不使用基类的查询集
        batch_size=opt.episodes_per_batch,  # 每批处理的episode数量
        epoch_size=opt.episodes_per_batch * 1000,  # 每轮训练的总episode数 = 批大小 × 1000
        num_workers=4  # 数据加载的并行进程数
    )

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.test_way,
        nKbase=0,
        nSupport=opt.test_shot,  # num training examples per novel category
        nQueryNovel=opt.test_query * opt.test_way,  # num test examples for all the novel categories
        nQueryBase=0,  # num test examples for all the base categories
        batch_size=1,  # 每个 batch 一个 episode（验证时一般不做多个并行任务）
        num_workers=0,
        epoch_size=1 * opt.test_episode,  # num of batches per epoch   	#验证时一次评估固定次数
    )

    set_gpu(opt.gpu)  # 设置使用的GPU
    check_dir('./experiments/')  # 创建实验目录
    check_dir(opt.save_path)  # 创建保存模型的目录

    log_file_path = os.path.join(opt.save_path, "train_log.txt")  # 构建日志文件路径
    log(log_file_path, str(vars(opt)))  # 记录配置文件中的参数信息到日志文件

    # (embedding_net, cls_head) = get_model(opt)  # 获取模型的结构   返回embedding_net（嵌入网络）和 cls_head（分类头） ！！！！
    # add_te_func = get_task_embedding_func(opt)  # 获取任务嵌入函数   ！！！！
    # postprocessing_net = get_postprocessing_model(opt)  # 获取后处理网络   对模型的输出结果

#     # 优化器
#     optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
#                                  {'params': cls_head.parameters()},
#                                  {'params': add_te_func.parameters()},
#                                  {'params': postprocessing_net.parameters()}],
#                                 lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
#
# # Load saved model checkpoints   加载保存的模型检查点
#     if opt.load is not None:
#         saved_models = torch.load(opt.load)    #用于恢复训练的进度
#         # NOTE: there is a `-1` because `epoch` starts counting from 1
#         last_epoch = saved_models['epoch'] - 1 if 'epoch' in saved_models.keys() else -1   #更新学习率调度器的状态
#         embedding_net.load_state_dict(saved_models['embedding'])    #恢复嵌入网络
#         cls_head.load_state_dict(saved_models['head'])      #恢复分类头
#         if 'task_embedding' in saved_models.keys():             #恢复状态字典也会加载
#             add_te_func.load_state_dict(saved_models['task_embedding'])
#         if 'postprocessing' in saved_models.keys():
#             postprocessing_net.load_state_dict(saved_models['postprocessing'])
#         if 'optimizer' in saved_models.keys():
#             optimizer.load_state_dict(saved_models['optimizer'])
#     else:
#         last_epoch = -1
#
#
#     # 学习率调度器
#     #学习率随着训练周期的变化规则
#     #前20周期使用学习率为1
#     #20-40周期使用学习率为0.06
#     #40-50周期使用学习率为0.012
#     #50-60周期使用学习率为0.0024
#     lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
#
#     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
#                                                      lr_lambda=lambda_epoch,
#                                                      last_epoch=last_epoch)
#
#     max_val_acc = 0.0
#
#     timer = Timer()
#     x_entropy = torch.nn.CrossEntropyLoss()
