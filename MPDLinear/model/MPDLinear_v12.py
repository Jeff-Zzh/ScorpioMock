import torch
import torch.nn as nn

from MPDLinear.layer.AttentionLayer import AttentionLayer

'''
MPDLinear_v12: 包含module1, module2, module4
'''
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


class multi_part_series_decompose(nn.Module):
    """
    多成分序列分解模块
    Series decomposition block
    将输入的时间序列分解为多种成分(趋势、季节、周期性，残差、非线性趋势成分)
    """

    def __init__(self, kernel_size, seq_len, channels):
        super(multi_part_series_decompose, self).__init__()
        # stride=1 表示窗口每次只移动一个时间步，这意味着每一个时间步都会有一个对应的移动平均值计算结果,计算每个时间步的移动平均值
        # 移动平均的结果会是一个平滑后的时间序列，去除了高频的短期波动，从而突出了低频的长期趋势
        self.moving_avg = moving_avg(kernel_size, stride=1) # 移动平均法,平滑时间序列数据,它通过在一个固定大小的窗口内对数据进行平均，来消除短期波动，从而突出数据中的长期趋势
        # self.periodic = moving_avg(kernel_size // 2, stride=1)

        # 双重非线性映射-捕捉时间序列中的复杂非线性趋势
        # 处理具有复杂多维关系的时间序列数据非常有用，能够增强模型对非线性趋势的建模能力
        self.nonlinear_trend_seq_len = nn.Sequential( # 非线性趋势 提取模块：从时间序列的趋势成分中提取出可能存在的非线性模式，对于具有复杂非线性趋势的时间序列数据，这种方法可以更好地捕捉数据中的非线性变化趋势
            # 在 seq_len 维度进行线性变换（即，每个channel的多个时间步seq_len进行非线性映射）
            nn.Linear(seq_len, seq_len, bias=True), # 线性层（全连接层）
            nn.ReLU(), # 应用非线性激活函数，使得模型能够拟合更加复杂的非线性关系
            nn.Linear(seq_len, seq_len, bias=True) # 线性层
        )  # 非线性趋势成分
        self.nonlinear_trend_channels = nn.Sequential(
            # 在 channels 维度进行线性变换（即，每个 seq_len 的多个特征 channels 进行非线性映射）
            nn.Linear(channels, channels, bias=True),  # 线性层（全连接层）
            nn.ReLU(),  # 应用非线性激活函数，使得模型能够拟合更加复杂的非线性关系
            nn.Linear(channels, channels, bias=True)  # 线性层
        )  # 非线性趋势成分

    def get_periodic_component(self, x):
        '''
        使用傅里叶变换，将时间域（time domain）的信号转换到频域（frequency domain），从而捕捉信号的周期性特征
        在频域中，周期性成分会以显著的频率成分体现出来，精确地捕捉到周期性的模式
        :param x: 输入形状[batch_size, seq_len, channels]
        :return:
        '''
        # 对输入的时间序列 进行 快速傅里叶变换（FFT），提取周期性成分
        fft_result = torch.fft.fft(x, dim=1) # 在张量 x 的第1维度（时间步序列seq_len的维度上）上进行傅里叶变换, 捕捉序列随时间变化的周期性特征
        # 设置阈值，去除高频噪声，仅保留主要周期性成分；保留低频分量以提取主要的周期性成分，去除高频噪声，seq_len维度
        # threshold 是保留的低频分量的阈值,是 seq_len 的 10%，即保留低频分量的前 10%，剩余的 90% 被视为高频噪声
        threshold = int(x.size(1) * 0.1) # 保留的低频分量数量:int(30*0.1)=3;在训练模型过程中，可以对threshold做调整，以确保提取到的周期性成分能够最佳地表示数据中的周期性特征
        fft_result[:, threshold:] = 0 # 去除高频噪声: 将超过阈值的高频分量置为 0，即去除了高频噪声，只保留主要的低频周期性成分
        # 对保留了低频周期性成分的分量的 fft_result 进行逆傅里叶变换，将数据从频域转换回时间域
        periodic_series = torch.fft.ifft(fft_result, dim=1).real
        return periodic_series

    def forward(self, x):
        '''
        :param x: 模型输入时间序列 , 形状：[batch_size, seq_len, channels]
        :return: 多元分解序列list n * [batch_size, seq_len, channels]
        '''
        multi_part_series = [] # 多元分解序列list
        # 1. 趋势序列成分
        trend_series = self.moving_avg(x) # 趋势成分（Trend Component） [batch_size, seq_len, channels]

        # 2. 非线性趋势序列成分 (从趋势序列中提取非线性趋势序列)
        trend_series_permuted = trend_series.permute(0, 2, 1)  # 输出[batch_size, channels, seq_len]
        # 先在 seq_len 维度上提取非线性趋势序列，进行非线性映射
        nonlinear_trend_series = self.nonlinear_trend_seq_len(trend_series_permuted)
        nonlinear_trend_series = nonlinear_trend_series.permute(0, 2, 1) # 输出[batch_size, seq_len, channels]
        # 再从 nonlinear_trend_series 中提取 在 channels 维度上提取非线性趋势序列，进行非线性映射
        # 能够捕捉到跨维度的非线性关系，能够充分建模多维度之间复杂依赖关系，更适合处理数据中高度复杂和交互性强的模式
        nonlinear_trend_series = self.nonlinear_trend_channels(nonlinear_trend_series)  # 非线性趋势 成分

        # 3.季节趋势成分
        res_trend_series = x - trend_series # 去除趋势序列后的残留序列（residual series），res_trend_series 序列主要包含季节性成分、周期性成分及噪声
        seasonal_series = self.moving_avg(res_trend_series) # 季节性成分（Seasonal Component）

        # 4.周期趋势成分
        res_trend_seasonal_series = res_trend_series - seasonal_series # 去除季节性成分后的残留序列（Residual series）
        periodic_series = self.get_periodic_component(res_trend_seasonal_series)  # 取出周期性成分（Periodic Component） 使用移动平均法/傅里叶变换/希尔伯特-黄变换

        # 5.残差噪声趋势成分
        res_trend_season_periodic_series = res_trend_seasonal_series - periodic_series # 残差（噪声） 成分

        multi_part_series = [trend_series, seasonal_series, periodic_series, res_trend_season_periodic_series, nonlinear_trend_series]

        # 返回多个成分（序列）
        return multi_part_series # [batch_size, seq_len, channels]

class MPDLinear_v12(nn.Module):
    """
    Decomposition-Linear
    基于时间序列分解的线性预测模型
    """

    def __init__(self, configs):
        super(MPDLinear_v12, self).__init__()
        self.seq_len = configs.seq_len  # 输入序列长度
        self.pred_len = configs.pred_len  # 预测序列长度
        self.individual = configs.individual  # 是否为每个通道单独应用线性模型
        self.channels = configs.enc_in  # 输入通道数（特征数feature数）

        # Decompsition Kernel Size
        kernel_size = 25  # 分解窗口大小
        self.decomposition = multi_part_series_decompose(kernel_size, self.seq_len, self.channels)  # 序列分解块实例


         # 所有通道 共享一个线性层
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.Linear_Periodic = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.Linear_Residual = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.Linear_Nonlinear = nn.Linear(self.seq_len, self.pred_len, bias=True)

        # 定义可自学习的序列权重向量，用于对多个序列进行加权求和，nn.Parameter
        self.series_weights_vector = None  # init放在forward中, 同时保证不会每次前向传播forward都会init series_weights_vector

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
        # x: [Batch, Input length, Channel] 即 [batch_size, seq_len, channels]

        # 获取多元分解序列
        multi_part_series = self.decomposition(x)  # 将输入的多元时间序列分解为 多个 成分
        series_num = len(multi_part_series)
        trend_series, seasonal_series, periodic_series, res_trend_season_periodic_series, nonlinear_trend_series = multi_part_series[:]  # [batch_size, seq_len, channels]

        # 动态初始化可自学习的序列权重向量，权重向量会在训练过程中通过反向传播进行学习和更新，通过梯度下降进行优化
        if self.series_weights_vector is None:  # 避免每次 forward 调用时重新初始化
            # 将张量注册为模型的可学习参数，使用 nn.Parameter 创建的张量将自动添加到模型的参数列表中，并且会在反向传播过程中更新其值
            # 将一个张量包裹在 nn.Parameter 中时，它会被视为模型的一个参数，自动加入到模型的参数列表中，并且会在反向传播时计算梯度，从而进行优化
            # 创建一个形状为 (series_num,) 的可学习权重向量。这个向量中的元素初始值为 1，并且它会在模型训练过程中通过梯度下降进行更新。
            # series_num 代表不同分解序列的数量，因此这个权重向量的每个元素对应一个分解序列的权重，训练过程中会学习这些权重，从而平衡不同分解序列在最终输出中的贡献
            self.series_weights_vector = nn.Parameter(torch.ones(series_num))  # 2 个 多元分解序列：趋势序列 和 季节序列

        # 调整张量形状：为了方便对每个通道应用线性层Linear，将输入张量的形状从[batch_size, seq_len, channels] 调整为 [batch_size, channel, seq_len]
        trend_init = trend_series.permute(0, 2,1)  # 重排张量维度：[batch_size, seq_len, channels] -> [batch_size, channels, seq_len]
        seasonal_init = seasonal_series.permute(0, 2, 1)
        periodic_init = periodic_series.permute(0, 2, 1)
        res_trend_season_periodic_init = res_trend_season_periodic_series.permute(0, 2, 1)
        nonlinear_trend_init = nonlinear_trend_series.permute(0, 2, 1)

        # 所有通道Channel共享一个线性模型，则直接对整个序列进行线性变换
        trend_output = self.Linear_Trend(trend_init)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        periodic_output = self.Linear_Periodic(periodic_init)
        res_trend_season_periodic_output = self.Linear_Residual(res_trend_season_periodic_init)
        nonlinear_trend_output = self.Linear_Nonlinear(nonlinear_trend_init)

        # 应用混合自注意力机制
        # 将时间维度和通道维度的注意力机制结合使用，使得模型不仅能够在通道之间动态调整权重，还能够在不同的时间步之间动态调整权重，进一步提升模型的复杂度和适应性
        # feature_attention的 QKV 的 dim_in和dim_out都是channels大小，输入形状从(batch_size,channels,pred_len)，调整为(batch_size,pred_len,channels)
        trend_output, trend_feature_att = self.feature_attention(
            trend_output.permute(0, 2, 1))  # 输出的trend_output的形状为[batch_size,pred_len,channels]
        # pred_len_temporal_attention的 QKV 的dim_in和dim_out都是pred_len大小，输入形状 由(batch_size,pred_len,channels) 调整为 (batch_size,channels,pred_len)
        # 输出形状为 (batch_size,channels,pred_len)
        trend_output, trend_temporal_att = self.pred_len_temporal_attention(trend_output.permute(0, 2, 1))

        seasonal_output, seasonal_feature_att = self.feature_attention(seasonal_output.permute(0, 2, 1))
        seasonal_output, seasonal_temporal_att = self.pred_len_temporal_attention(seasonal_output.permute(0, 2, 1))

        periodic_output, periodic_feature_att = self.feature_attention(periodic_output.permute(0, 2, 1))
        periodic_output, periodic_temporal_att = self.pred_len_temporal_attention(periodic_output.permute(0, 2, 1))

        res_trend_season_periodic_output, res_feature_att = self.feature_attention(
            res_trend_season_periodic_output.permute(0, 2, 1))
        res_trend_season_periodic_output, res_temporal_att = self.pred_len_temporal_attention(
            res_trend_season_periodic_output.permute(0, 2, 1))

        nonlinear_trend_output, nonlinear_feature_att = self.feature_attention(nonlinear_trend_output.permute(0, 2, 1))
        nonlinear_trend_output, nonlinear_temporal_att = self.pred_len_temporal_attention(
            nonlinear_trend_output.permute(0, 2, 1))

        forward_output = (  # [batch_size, channel, pred_len]
                self.series_weights_vector[0] * trend_output +  # 将output tensor的每一个元素都与数  self.series_weights_vector[0] 相乘
                self.series_weights_vector[1] * seasonal_output +
                self.series_weights_vector[2] * periodic_output +
                self.series_weights_vector[3] * res_trend_season_periodic_output +
                self.series_weights_vector[4] * nonlinear_trend_output
        )
        return forward_output.permute(0, 2, 1)  # to [Batch, Output length, Channel]即 [batch_size, pred_len, channel] 这就是outputs = model(inputs) 中outputs的形状
