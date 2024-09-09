import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from MPDLinear.layer.AttentionLayer import AttentionLayer

'''
MPDLinear_v14: 包含 module2 , module3, module4
'''

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

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
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MPDLinear_v14(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(MPDLinear_v14, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        # 定义可自学习的序列权重向量，用于对多个序列进行加权求和，nn.Parameter
        self.series_weights_vector = None  # init放在forward中, 同时保证不会每次前向传播forward都会init series_weights_vector

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        # 添加注意力层
        # 特征的自注意力机制层:应用于通道（channels）维度，计算不同特征通道之间的注意力权重
        self.feature_attention = AttentionLayer(dim_in=self.channels, dim_out=self.channels)
        # 时间的自注意力机制层1: 应用于输入时间步 seq_len 维度，计算不同时间步之间的注意力权重
        self.seq_len_temporal_attention = AttentionLayer(dim_in=self.seq_len, dim_out=self.seq_len)
        # 时间的自注意力机制层2: 应用于输出（预测）时间步 pred_len 维度，计算不同时间步之间的注意力权重
        self.pred_len_temporal_attention = AttentionLayer(dim_in=self.pred_len, dim_out=self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        # 动态初始化可自学习的序列权重向量，权重向量会在训练过程中通过反向传播进行学习和更新，通过梯度下降进行优化
        if self.series_weights_vector is None:  # 避免每次 forward 调用时重新初始化
            # 将张量注册为模型的可学习参数，使用 nn.Parameter 创建的张量将自动添加到模型的参数列表中，并且会在反向传播过程中更新其值
            # 将一个张量包裹在 nn.Parameter 中时，它会被视为模型的一个参数，自动加入到模型的参数列表中，并且会在反向传播时计算梯度，从而进行优化
            # 创建一个形状为 (series_num,) 的可学习权重向量。这个向量中的元素初始值为 1，并且它会在模型训练过程中通过梯度下降进行更新。
            # series_num 代表不同分解序列的数量，因此这个权重向量的每个元素对应一个分解序列的权重，训练过程中会学习这些权重，从而平衡不同分解序列在最终输出中的贡献
            self.series_weights_vector = nn.Parameter(torch.ones(2))  # 2 个 多元分解序列：趋势序列 和 季节序列

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # 应用混合自注意力机制
        trend_output, trend_feature_att = self.feature_attention(trend_output.permute(0, 2, 1))  # 输出的trend_output的形状为[batch_size,pred_len,channels]
        # pred_len_temporal_attention的 QKV 的dim_in和dim_out都是pred_len大小，输入形状 由(batch_size,pred_len,channels) 调整为 (batch_size,channels,pred_len)
        # 输出形状为 (batch_size,channels,pred_len)
        trend_output, trend_temporal_att = self.pred_len_temporal_attention(trend_output.permute(0, 2, 1))

        seasonal_output, seasonal_feature_att = self.feature_attention(seasonal_output.permute(0, 2, 1))
        seasonal_output, seasonal_temporal_att = self.pred_len_temporal_attention(seasonal_output.permute(0, 2, 1))

        forward_output = self.series_weights_vector[0] * trend_output + self.series_weights_vector[1] * seasonal_output
        return forward_output.permute(0, 2, 1)  # to [Batch, Output length, Channel]
