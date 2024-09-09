import torch
import torch.nn as nn
from MPDLinear.config.ModelConfig import ModelConfig
from MPDLinear.layer.AttentionLayer import AttentionLayer

'''
MPDLinear_v6:包含module1和module3
'''

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    移动平均块，用于平滑时间序列数据，以突出显示数据的趋势成分。
    移动平均是一种常用的时间序列平滑技术，可以 ”消除短期波动，保留长期趋势“
    """

    def __init__(self, kernel_size, stride):
        '''
        移动平均的窗口大小
        移动平均的步长
        '''
        super(moving_avg, self).__init__()
        # kernel_size 是一个整数，表示滑动窗口的大小，也就是在计算移动平均时，需要包含的连续时间步的数量
        # 例如，如果 kernel_size=3，那么在计算某个时间步的平均值时，会考虑该时间步及其前后各一个时间步的数据
        self.kernel_size = kernel_size
        # stride 是一个整数，表示滑动窗口在时间序列上移动的步长
        # stride=1 表示窗口每次只移动一个时间步，这意味着每一个时间步都会有一个对应的移动平均值计算结果
        # 如果 stride=2，则窗口每次移动两个时间步，这样会跳过一些时间步，减少计算的移动平均值数量
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2) # 序列自动填充 进行一维卷积操作（即移动平均）
        # self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) # 序列前后 手动填充

    def forward(self, x):
        # padding on the both ends of time series 在时间序列的两端进行填充，以确保边界时间步也能计算平均值
        # front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) # 在序列的前端进行填充，填充的内容是第一个时间步的值，填充长度为 (self.kernel_size - 1) // 2
        # end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1) # 在序列的末端进行填充，填充的内容是最后一个时间步的值，填充长度同样为 (self.kernel_size - 1) // 2
        # x = torch.cat([front, x, end], dim=1)
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


class MPDLinear_v6(nn.Module):
    """
    Decomposition-Linear
    基于时间序列分解的线性预测模型
    """

    def __init__(self, configs:ModelConfig):
        super(MPDLinear_v6, self).__init__()
        self.seq_len = configs.seq_len  # 输入序列长度
        self.pred_len = configs.pred_len  # 预测序列长度
        self.individual = configs.individual  # 是否为每个通道单独应用线性模型
        self.channels = configs.enc_in  # 输入通道数（特征数feature数）

        # Decompsition Kernel Size
        kernel_size = configs.decomposition_kernel_size  # 分解窗口大小 默认25
        self.decomposition = multi_part_series_decompose(kernel_size, self.seq_len, self.channels)  # 序列分解块实例

        if self.individual:
            self.Linear_Trend = nn.ModuleList() # 存储多个nn.Linear层
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Periodic = nn.ModuleList()
            self.Linear_Residual = nn.ModuleList()
            self.Linear_Nonlinear = nn.ModuleList()

            for i in range(self.channels): # 为每个channel创建属于这个channel自己的Linear层
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len, bias=True)) # nn.Linear定义一个神经网络的线性层，做了一个线性变换y = x*w + b (输入神经元个数，输出神经元个数，偏置) 默认设置偏置为True
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len, bias=True))
                self.Linear_Periodic.append(nn.Linear(self.seq_len, self.pred_len, bias=True))
                self.Linear_Residual.append(nn.Linear(self.seq_len, self.pred_len, bias=True))
                self.Linear_Nonlinear.append(nn.Linear(self.seq_len, self.pred_len, bias=True))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:  # 所有通道 共享一个线性层
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len, bias=True)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len, bias=True)
            self.Linear_Periodic = nn.Linear(self.seq_len, self.pred_len, bias=True)
            self.Linear_Residual = nn.Linear(self.seq_len, self.pred_len, bias=True)
            self.Linear_Nonlinear = nn.Linear(self.seq_len, self.pred_len, bias=True)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel] 即 [batch_size, seq_len, channels]

        # 获取多元分解序列
        multi_part_series = self.decomposition(x)  # 将输入的多元时间序列分解为 多个 成分
        series_num = len(multi_part_series)
        trend_series, seasonal_series, periodic_series, res_trend_season_periodic_series, nonlinear_trend_series = multi_part_series[:] # [batch_size, seq_len, channels]


        # 调整张量形状：为了方便对每个通道应用线性层Linear，将输入张量的形状从[batch_size, seq_len, channels] 调整为 [batch_size, channel, seq_len]
        trend_init = trend_series.permute(0, 2, 1)  # 重排张量维度：[batch_size, seq_len, channels] -> [batch_size, channels, seq_len]
        seasonal_init = seasonal_series.permute(0, 2, 1)
        periodic_init = periodic_series.permute(0, 2, 1)
        res_trend_season_periodic_init = res_trend_season_periodic_series.permute(0, 2, 1)
        nonlinear_trend_init = nonlinear_trend_series.permute(0, 2, 1)

        if self.individual:
            # 为每个通道创建输出张量,存储每个通道的线性层输出结果 shape:[batch_size, channel, pred_len]
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            periodic_output = torch.zeros([periodic_init.size(0), periodic_init.size(1), self.pred_len],
                                          dtype=periodic_init.dtype).to(periodic_init.device)
            res_trend_season_periodic_output = torch.zeros([res_trend_season_periodic_init.size(0), res_trend_season_periodic_init.size(1), self.pred_len],
                                                           dtype=res_trend_season_periodic_init.dtype).to(res_trend_season_periodic_init.device)
            nonlinear_trend_output = torch.zeros([nonlinear_trend_init.size(0), nonlinear_trend_init.size(1), self.pred_len],
                                                 dtype=nonlinear_trend_init.dtype).to(nonlinear_trend_init.device)

            # 如果为每个通道单独应用线性模型，则遍历每个通道Channel，对趋势和季节性成分分别应用线性模型，并将结果存储在 seasonal_output 和 trend_output 中
            for i in range(self.channels):
                # 该通道的季节性成分经过线性层后的输出
                # 选取 seasonal_init 张量中所有批次（batch_size 维度）的第 i 个通道的数据，并且选择该通道中所有的时间步（seq_len 维度）
                # 返回的张量形状 seasonal_init [batch_size, channels, seq_len] -> seasonal_init[:, i, :] 形状[batch_size, seq_len] ->经过线性层 [batch_size, pred_len]
                # 对每个样本（每一个batch），提取了该batch第 i 个通道的整个时间序列，然后给Linear层输入
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :]) # 第i个channel的Linear层 形状为[batch_size, seq_len] -> [batch_size, pred_len]
                # 该通道的趋势性成分经过线性层后的输出
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
                # 周期趋势成分
                periodic_output[:, i, :] = self.Linear_Periodic[i](periodic_init[:, i, :])
                # 残差趋势成分
                res_trend_season_periodic_output[:, i, :] = self.Linear_Residual[i](res_trend_season_periodic_init[:, i, :])
                # 非线性趋势成分
                nonlinear_trend_output[:, i, :] = self.Linear_Nonlinear[i](nonlinear_trend_init[:, i, :])
        else: # 如果所有通道Channel共享一个线性模型，则直接对整个序列进行线性变换
            # output的形状均为： [batch_size, channels, seq_len] -> [batch_size, channels, pred_len]
            trend_output = self.Linear_Trend(trend_init)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            periodic_output = self.Linear_Periodic(periodic_init)
            res_trend_season_periodic_output = self.Linear_Residual(res_trend_season_periodic_init)
            nonlinear_trend_output = self.Linear_Nonlinear(nonlinear_trend_init)


        forward_output = trend_output + seasonal_output + periodic_output + res_trend_season_periodic_output + nonlinear_trend_output
        return forward_output.permute(0, 2, 1)  # to [Batch, Output length, Channel]即 [batch_size, pred_len, channel] 这就是outputs = model(inputs) 中outputs的形状
