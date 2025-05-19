# encoding: utf-8
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
# from fvcore.nn import FlopCountAnalysis

from model.actionlet import Actlet
from torchlight import import_class
import copy
import random


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


# 梯度计算
class TemporaryGrad(object):
    # 利用with TemporaryGrad()会自动调用
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        # 获取是否启用梯度计算
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)


class Progress(nn.Module):
    def __init__(self, base_encoder=None,
                 mode='train',
                 num_class=60,
                 num_point=25,
                 num_person=2,
                 graph=None,
                 graph_args=dict(),
                 in_channels=3,
                 drop_out=0,block_index=None ):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.encoder1 = base_encoder(graph=graph, graph_args=graph_args,
                                     num_class=num_class, num_point=num_point,
                                     num_person=num_person,
                                     in_channels=3,
                                    )
        self.encoder2 = base_encoder(graph=graph, graph_args=graph_args,
                                     num_class=num_class, num_point=num_point,
                                     num_person=num_person,
                                     )
        self.encoder3 = base_encoder(graph=graph, graph_args=graph_args,
                                     num_class=num_class, num_point=num_point,
                                     num_person=num_person,
                                     )
        self.encoder4 = base_encoder(graph=graph, graph_args=graph_args,
                                     num_class=num_class, num_point=num_point,
                                     num_person=num_person,
                                     )


        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        # actclr
        self.mode = mode
        self.actionlet1 = Actlet(in_features=64,out_features=64,mode=self.mode)
        self.actionlet2 = Actlet(in_features=128,out_features=128, mode=self.mode)
        self.actionlet3 = Actlet(in_features=256,out_features=256,mode=self.mode)

        base_channel = 64
        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, block_index=None):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x_mean = x.mean(dim=0, keepdim=True)   #average motion
        n,c,t,v,m = x_mean.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x_mean = x_mean.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        x_mean = self.data_bn(x_mean)
        x_mean = x_mean.view(n,c,t,v,m).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)

        x = self.encoder1(x, block_index=1)    #h(x)
        x_mean = self.encoder1(x_mean,block_index=1)

        act_out = self.actionlet1(x,x_mean,n,m,N,M)   #act(h(x))  权重
        x = x * act_out
        x = self.encoder2(x, block_index=2)


        x = self.encoder3(x, block_index=3)

        x = self.encoder4(x, block_index=4)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)

    # class Progress(nn.Module):
    #     def __init__(self, base_encoder=None,
    #                  mode='train',
    #                  num_class=60,
    #                  num_point=25,
    #                  num_person=2,
    #                  graph=None,
    #                  graph_args=dict(),
    #                  in_channels=3, drop_out=0):
    #         """
    #         K: queue size; number of negative keys (default: 32768)
    #         m: momentum of updating key encoder (default: 0.999)
    #         T: softmax temperature (default: 0.07)
    #         """
    #         super().__init__()
    #         base_encoder = import_class(base_encoder)
    #         self.encoder1 = base_encoder(graph=graph, graph_args=graph_args,
    #                                      num_class=num_class, num_point=num_point,
    #                                      num_person=num_person,
    #                                      in_channels=3, out_channels=64, first=True, last=False)
    #         self.encoder2 = base_encoder(graph=graph, graph_args=graph_args,
    #                                      num_class=num_class, num_point=num_point,
    #                                      num_person=num_person,
    #                                      in_channels=64, out_channels=128, first=False, last=False)
    #         self.encoder3 = base_encoder(graph=graph, graph_args=graph_args,
    #                                      num_class=num_class, num_point=num_point,
    #                                      num_person=num_person,
    #                                      in_channels=128, out_channels=256, first=False, last=False)
    #         self.last_layer = last_layer(graph=graph, graph_args=graph_args,
    #                                      num_class=num_class, num_point=num_point,
    #                                      num_person=num_person,
    #                                      in_channels=256, out_channels=256, first=False, last=True)
    #
    #         self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
    #         # actclr
    #         self.mode = mode
    #         self.actionlet1 = Actlet(in_features=64, out_features=64, mode=self.mode)
    #         self.actionlet2 = Actlet(in_features=128, out_features=128, mode=self.mode)
    #         self.actionlet3 = Actlet(in_features=256, out_features=256, mode=self.mode)
    #
    #         base_channel = 64
    #         self.fc = nn.Linear(base_channel * 4, num_class)
    #         nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
    #         bn_init(self.data_bn, 1)
    #         if drop_out:
    #             self.drop_out = nn.Dropout(drop_out)
    #         else:
    #             self.drop_out = lambda x: x
    # def forward(self, x):
    #     N, C, T, V, M = x.size()  #原数据
    #     x_mean = x.mean(dim=0, keepdim=True)   #average motion
    #     n,c,t,v,m = x_mean.size()
    #     x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
    #     x = self.data_bn(x)
    #     x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
    #     y = x.clone()
    #     x_mean = x_mean.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
    #     x_mean = self.data_bn(x_mean)
    #     x_mean = x_mean.view(n,c,t,v,m).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
    #
    #     x = self.encoder1(x)    #h(x)
    #     x_mean = self.encoder1(x_mean)
    #     act_out = self.actionlet1(x,x_mean,n,m,N,M)   #act(h(x))  权重
    #     #todo 考虑融合方式
    #     x1 = x * act_out + x
    #
    #     x = self.encoder2(x1)
    #     x_mean = self.encoder2(x_mean)
    #     act_out = self.actionlet2(x, x_mean,n,m,N,M)
    #     x2 = x * act_out + x
    #
    #     x = self.encoder3(x2)
    #     x_mean = self.encoder3(x_mean)
    #     act_out = self.actionlet3(x, x_mean,n,m,N,M)
    #     x3 = x * act_out + x
    #
    #     out = self.last_layer(x3)
    #     # N*M,C,T,V
    #     c_new = out.size(1)
    #     x = out.view(N, M, c_new, -1)
    #     x = x.mean(3).mean(1)
    #     x = self.drop_out(x)
    #
    #     return self.fc(x)
    #






