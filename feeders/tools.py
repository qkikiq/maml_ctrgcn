import random
import matplotlib.pyplot as plt
import numpy as np
import pdb

import torch
import torch.nn.functional as F

def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    #valid_frame_num : 有效帧数
    #p_interval : 裁剪比例区间
    #window : 输出帧数
    C, T, V, M = data_numpy.shape  # C: 通道数, T: 时间步长, V: 关节点数, M: 人数
    begin = 0  #起始帧索引
    end = valid_frame_num  #终止帧索引
    valid_size = end - begin  #有效帧数

    #crop
    if len(p_interval) == 1:    # 固定裁剪比例 如果只有一个值，执行中心裁剪
        p = p_interval[0]  #取固定的裁剪比例
        bias = int((1-p) * valid_size/2)   #计算裁剪的起始偏移
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop 中心裁剪
        cropped_length = data.shape[1]
    else:   #如果有两个值，执行随机裁剪
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]   #生成随即裁剪比例
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
        #裁剪长度限制在64到有效帧数之间
        bias = np.random.randint(0,valid_size-cropped_length+1)  #裁剪的起始位置
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]   # 按照计算结果裁剪时间步长
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize  重采样：标准化输入数据的大小
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    # 使用双线性插值调整时间维度到目标 window 大小，可进行上/下采样
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


import numpy as np
import random


def auto_pading(data_numpy, size, random_pad=False):
    # 获取输入数据的形状：C: 通道数，T: 时间步长，V: 骨架关键点数，M: 样本数
    C, T, V, M = data_numpy.shape

    # 如果时间步长 T 小于目标大小 size
    if T < size:
        # 如果 random_pad 为 True，随机选择填充的位置；否则从 0 开始填充
        begin = random.randint(0, size - T) if random_pad else 0

        # 创建一个全零的目标数组，其形状为 (C, size, V, M)，即填充后的数据形状
        data_numpy_paded = np.zeros((C, size, V, M))

        # 将原始数据填充到目标数组中，位置从 `begin` 开始
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy

        # 返回填充后的数据
        return data_numpy_paded
    else:
        # 如果时间步长 T 已经大于或等于 size，直接返回原始数据
        return data_numpy


import random


def random_choose(data_numpy, size, auto_pad=True):
    # input: C, T, V, M，C: 通道数，T: 时间步长，V: 骨架关键点数，M: 样本数
    # 随机选择时间维度T的一段（大小为size）。如果T小于size，则进行填充。

    # 获取输入数据的形状：C, T, V, M
    C, T, V, M = data_numpy.shape

    # 如果输入数据的时间步长 T 等于目标大小 size，直接返回原始数据
    if T == size:
        return data_numpy

    # 如果输入数据的时间步长 T 小于目标大小 size
    elif T < size:
        # 如果 auto_pad 为 True，则对数据进行填充
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            # 如果 auto_pad 为 False，直接返回原始数据
            return data_numpy

    # 如果输入数据的时间步长 T 大于目标大小 size
    else:
        # 从 0 到 (T - size) 之间随机选择一个起始位置 begin
        begin = random.randint(0, T - size)

        # 从原始数据中选择从 begin 开始，长度为 size 的片段
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3  每个时间步长的正余弦
    zeros = torch.zeros(rot.shape[0], 1)  # T,1   用于填充的零元素
    ones = torch.ones(rot.shape[0], 1)  # T,1     构造单位元素

    #计算x轴旋转矩阵
    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    #计算y轴旋转矩阵
    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    #计算z轴旋转矩阵
    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)
    #先
    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)  #随机旋转角度向量
    rot = torch.stack([rot, ] * T, dim=0)    #将旋转向量扩展到时间维度
    rot = _rot(rot)  # T,3,3   #每个时间步对应一个3*3旋转矩阵
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch

def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy
