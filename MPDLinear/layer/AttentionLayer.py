import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    '''
    实现缩放点积注意力机制
    让模型在处理序列数据时，动态地关注序列中的不同部分，从而更有效地提取信息
    '''
    def __init__(self, dim_in, dim_out):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(dim_in, dim_out) # 初始化时规定Linear层输入输出形状
        self.key = nn.Linear(dim_in, dim_out)
        self.value = nn.Linear(dim_in, dim_out)
        # 引入缩放因子，可以防止 softmax 的输入值变得过大，导致梯度消失或模型过于集中注意
        self.scale = 1.0 / np.sqrt(dim_out) # 缩放因子，根号下d_k, 对 dim_out 进行平方根计算。如果 dim_out 很大，平方根会将其缩小，使得点积的结果在较小的范围内分布

    def forward(self, x):
        # x: [batch_size, seq_len/pred_len, channels]特征自注意力机制层 or [batch_size, channels, seq_len/pred_len]输入/输出时间步自注意力机制层
        # Q K V 输出形状均为  [batch_size, seq_len/pred_len, channels]特征自注意力机制层 or [batch_size, channels, seq_len/pred_len]输入/输出时间步自注意力机制层
        Q = self.query(x) # 模型在输入中想要寻找的内容
        K = self.key(x) # 输入中可以用来匹配查询的特征
        V = self.value(x) # 输入数据中实际要提取的信息，通常是最终要加权求和的内容

        # 计算注意力分数 attention_scores = (Q · K^T) / d_k^1/2
        # 1.K.transpose(-2, -1)：将 K 矩阵的最后两个维度进行转置，[batch_size, seq_len, dim_out] -> [batch_size, dim_out, seq_len] or [batch_size, channels, dim_out] -> [batch_size, dim_out, channels]
        # 转置K的原因：为了进行矩阵乘法 Q · K^T，需要确保 Q 和 K 的维度能够匹配
        # 2.torch.matmul(Q, K.transpose(-2, -1))：矩阵乘法matrix multiplication
        # Q:[batch_size, seq_len, dim_out] K^T:[batch_size, dim_out, seq_len], 运算后attention_scores: [batch_size, seq_len, seq_len] or [batch_size, channels, channels]
        # 对于每个批次中的元素，执行查询（Q）和转置后的键（K）的矩阵乘法，得到注意力分数矩阵。这个矩阵表示序列中每个位置与其他位置的相关性
        # 3.* self.scale: 对注意力分数进行缩放，防止点积值过大，导致在后续的 softmax 操作中梯度消失
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # 应用 Softmax 函数，得到注意力权重（Attention Weights）:表示在每个查询位置（Query）上，模型对不同键位置（Key）的关注度。这些权重将用于加权组合值（Value），从而生成最终的输出, 这些权重表示了在当前查询下，每个键的重要性
        # 1.F.softmax(attention_scores, dim=-1)：将原始的注意力分数转换为概率分布，使得每个位置的权重和为1，dim=-1：在最后一个维度上应用 softmax
        # 对于序列中的每个查询位置，softmax 将注意力分数转化为对所有键位置的概率分布。这意味着对于每个查询，该位置对其他位置的关注度被转换为一个概率值，所有的概率值之和为1
        attention_weights = F.softmax(attention_scores, dim=-1) # attention_weights 的形状与 attention_scores 形状相同：[batch_size, seq_len, seq_len] or [batch_size, channels, channels]

        # 应用注意力权重到 V (矩阵乘法):通过注意力权重将不同位置的信息加权求和，从而得到更为聚焦的特征表示
        # 通过使用注意力权重，模型可以在处理序列数据时，动态地选择和强调那些与当前任务相关的重要部分，而忽略不重要的信息
        # [batch_size, seq_len, seq_len] or [batch_size, channels, channels] * [batch_size, seq_len, dim_out] or [batch_size, channels, dim_out]
        # out形状为 [batch_size, seq_len, dim_out(channels)] or [batch_size, channels, dim_out(seq_len/pred_len)]
        out = torch.matmul(attention_weights, V)
        return out, attention_weights
