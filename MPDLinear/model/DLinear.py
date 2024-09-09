import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    移动平均块，用于平滑时间序列数据，以突出显示数据的趋势成分。
    移动平均是一种常用的时间序列平滑技术，可以去除短期波动，保留长期趋势
    """

    def __init__(self, kernel_size, stride):
        '''
        移动平均的窗口大小
        移动平均的步长
        '''
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) # 进行一维卷积操作（即移动平均）

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    将时间序列分解为趋势成分和季节性成分
    趋势成分由移动平均得到，季节性成分则是原始序列与趋势成分之间的差异
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        '''
        x：输入序列
        moving_mean：趋势成分（移动平均 ）
        res：季节性成分 （原始时间序列减去趋势成分）
        '''
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear
    基于时间序列分解的线性预测模型
    """

    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len  # 输入序列长度
        self.pred_len = configs.pred_len  # 预测序列长度

        # Decompsition Kernel Size
        kernel_size = 25  # 分解窗口大小
        self.decomposition = series_decomp(kernel_size)  # 序列分解块实例
        self.individual = configs.individual  # 是否为每个通道单独应用线性模型
        self.channels = configs.enc_in  # 输入通道数（特征数feature数）

         # 所有通道 共享一个线性层
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len, bias=True)

        # Use this two lines if you want to visualize the weights
        # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel] 即 [batch_size, seq_len, channels]
        seasonal_init, trend_init = self.decomposition(x)  # 将输入的多元时间序列分解为 趋势 和 季节性成分
        # 重排数据维度 适应线性层的输入格式，线性层通常期望输入的最后一维是特征维度feature（channels）。通过重新排列维度，确保了在处理每个时间步的特征时，能够更方便地应用线性层
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)  # 重新排列为 [Batch, Channel, Input length]。

        # 所有通道Channel共享一个线性模型，则直接对整个序列进行线性变换
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]

'''
这个模型的核心思想是将时间序列分解为趋势和季节性成分，然后分别对这些成分进行预测。
通过这种方式，模型能够更好地捕捉时间序列中的不同模式，从而提高预测的准确性。
'''