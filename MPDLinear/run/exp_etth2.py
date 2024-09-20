import platform
import time

import torch
import gc
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from MPDLinear.data_process.data_visualization import draw_all
from MPDLinear.model.MPDLinear_SOTA import MPDLinear_SOTA
from MPDLinear.model.model_dict import ModelDict
from MPDLinear.data_process.data_processor import preprocessing_data, load_data, clean_weather_and_save, \
    divide_data_by_geographic_and_save, feature_engineering, windows_select_single_label, \
    feature_engineering_for_electricity_and_save, feature_engineering_for_exchange_rate_and_save, \
    feature_engineering_for_etth1_and_save, feature_engineering_for_etth2_and_save
from MPDLinear.config.ModelConfig import ModelConfig
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import os
from datetime import datetime

from util.EarlyStopping import EarlyStopping
from util.logger import setup_logger
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler  # torch 1.9.0没有，torch2.4.1有,并且需要torch为CUDA版本的torch

# 定义不同的 seq_len 值列表(采样时间步间隔为hour)
seq_len_list = [24, 48, 72, 96, 120, 144, 168, 192, 336, 504, 672, 720]

# 配置
config = ModelConfig() # 初始化模型参数配置
dataset_path = '../datasets/9_LTSF_dataset/ori/ETTh2.csv'

# 加载数据
ori_dataset = load_data(dataset_path)

# 特征工程:进行时间特征提取
# 时间特征提取应发生在标准化/归一化之前,这是因为时间特征（例如 year、month、day、hour 等）是离散的或有明确的周期性，它们的意义不会因标准化而增强。如果先进行标准化，时间特征的原始信息（例如年份或月份的顺序和周期性）可能会被破坏，进而影响模型的性能
expand_time_feature_dataset = feature_engineering_for_etth2_and_save(ori_dataset)

# 选择数据集
dataset_name = 'ETTh2'
dataset_exp = expand_time_feature_dataset

# 查看基本信息
print("\n实验所选数据集的 数据集基础信息：")
print(dataset_exp.info())
# 显示经特征工程后的数据集的前几行数据
print("\n实验所选数据集的 数据集前几行数据：")
print(dataset_exp.head())

# 检查缺失值
print("\n查看 实验所选数据集的 数据缺失值情况：")
print(dataset_exp.isnull().sum())

# 填充缺失值(将所有缺失值替换为它之前最近的一个非缺失值,将前向填充后仍然存在的缺失值替换为它之后最近的一个非缺失值)
# 插值法利用数据的趋势和模式对缺失值进行填充，而前向填充和后向填充则确保了所有缺失值都能被有效填补。这样处理后的数据将更加完整，有助于提高后续数据分析和建模的准确性
# 先把数据集的所有列都转为数值列
dataset_exp = dataset_exp.apply(pd.to_numeric, errors='coerce')
# 强制将 'week_of_year'  UInt32 类型转换为 int64
dataset_exp['week_of_year'] = dataset_exp['week_of_year'].astype('int64')
# 先进行线性插值，处理连续数据的缺失值
dataset_exp.interpolate(method='linear', inplace=True)
# 然后使用前向填充和后向填充处理剩余的缺失值
dataset_exp.fillna(method='ffill', inplace=True)
dataset_exp.fillna(method='bfill', inplace=True)

# 检查填充后缺失值情况
print("\n插值/前向/后向填充后，实验所选数据集的缺失值情况：")
print(dataset_exp.isnull().sum())

# 特征选择/Label选择
# 选择数值型特征列，在时序预测任务中，日期和时间相关的特征（如 year、month、week、quarter 和 day）通常可以作为特征，
# 因为它们可以捕捉到数据的季节性和周期性变化。我们将这些列也作为特征列来构建模型。
selected_features = dataset_exp.columns[:14] # 选前14列作为feature

# 提取特征feature和选择预测目标变量target(用过去一段时间的所有特征来预测未来某一时间点的目标变量,可为temperature、humidity等等, 单目标预测，多目标预测)
X = dataset_exp[selected_features]  # 特征变量
y = dataset_exp['OT']  # 目标变量

# 对feature和label进行 标准化(数据均值为 0，标准差为 1) 或 归一化(所有值归一化到 [0, 1] 范围内)， 选其一，都进行数据存储
# Q:为什么需要对feature和label进行标准化 或 归一化？
# A:根据数据集的统计信息，特征列的取值范围和尺度差异较大（例如，某些特征的值在 0 到数万之间，而其他特征则在较小范围内变化）。这种情况下，标准化 或 归一化 是推荐的，以确保不同特征的值范围相似，从而帮助模型更好地收敛并避免某些特征对模型的影响过大。
# 标准化feature和target
# 标准化特征
scaler_X = StandardScaler()
X_standardized = scaler_X.fit_transform(X)
# 标准化目标变量
scaler_y = StandardScaler()
y_standardized = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten() # 将一维数组转换为二维数组 1 * 5 -> 5 * 1 然后再二维数组变为一维数组 1 * 5
# 将标准化后的数据转换为 DataFrame
X_feature_standardized_df = pd.DataFrame(X_standardized, columns=selected_features)
y_label_standardized_df = pd.DataFrame(y_standardized, columns=['OT'])
# 合并特征feature和目标变量target
standardized_data = pd.concat([X_feature_standardized_df, y_label_standardized_df], axis=1)
# 保存标准化后的数据到 CSV 文件
standardized_data.to_csv('../datasets/9_LTSF_dataset/processed/standardized/ETTh2_processed_standardized.csv', index=False)

# 归一化feature和target
# 归一化特征
scaler_X_minmax = MinMaxScaler()
X_normalized = scaler_X_minmax.fit_transform(X)
# 归一化目标变量
scaler_y_minmax = MinMaxScaler()
y_normalized = scaler_y_minmax.fit_transform(y.values.reshape(-1, 1)).flatten()
# 将归一化后的数据转换为 DataFrame
X_feature_normalized_df = pd.DataFrame(X_normalized, columns=selected_features)
y_label_normalized_df = pd.DataFrame(y_normalized, columns=['OT'])
# 合并特征和目标变量
normalized_data = pd.concat([X_feature_normalized_df, y_label_normalized_df], axis=1)
# 保存归一化后的数据到 CSV 文件
normalized_data.to_csv('../datasets/9_LTSF_dataset/processed/normalized/ETTh2_processed_normalized.csv', index=False)

print("已完成标准化 和 归一化，并已经存储标准化和归一化后的数据")

# 设置模型参数配置
# 选择模型
config.model = 'MPDLinear_SOTA'
config.batch_size = 64 # 数据集有1825个样本。通常，对于小型数据集，较小的 batch_size 会更合适，如 16、32(cpu) 或 64,128(gpu)。这样可以更充分地利用数据并防止内存溢出
# seq_len选择
# 常用起点: 可以从 30 天（一个月的时间序列数据）开始。这通常是一个合理的起点，既能捕捉短期趋势，又不会过长。
# 根据模型调整: 如果在训练过程中发现模型表现良好，可以逐步增加 seq_len，例如增加到 60 天、90 天或 180 天，看看是否能进一步提升模型性能
config.seq_len = None # [1, 3, 5, 7, 30, 90, 180, 365, 730]
# pred_len选择
# 推荐起点: 开始尝试 pred_len = 7（一周）或 pred_len = 10（十天），这些是常用的时间段，并且容易观察预测的准确性。
# 调整方向: 如果模型训练效果较好，你可以逐步增加 pred_len，如 14 天或 30 天，直到找到一个适合的平衡点。
config.pred_len = 5 # 1,3,5,7,10,30 数据集有 1825 个时间步，代表了按天采样5年的时间跨度。对于如此长时间的数据，如果 pred_len 设置得太大，模型可能难以学习到有效的长期依赖关系，因此需要谨慎选择。
config.individual = True
config.enc_in = len(selected_features) # 特征列数 = 通道数 = 14
config.num_epochs = 500
# EarlyStopping模块配置
config.es_patience = 5 # 100 跑全epoch(晚上跑)
config.es_verbose = True
config.es_delta = 0.00001
config.es_path = 'current_best_checkpoint.pt'
config.decomposition_kernel_size = 25
config.learning_rate = 0.0001 # 0.001 -> 0.0001 -> 0.00001
config.scaling_method = 'normalization' # 选择缩放方法 标准化/归一化 standardization/normalization
config.device = 'gpu'
config.dataset_name = dataset_name

device = None # torch所用设备
if config.device == 'gpu':
# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-----Using device: {device}-----")
else:
    device = torch.device('cpu')
    print(f"-----Using device: {device}-----")

for seq_len in seq_len_list:
    config.seq_len = seq_len  # 设置当前的 seq_len
    # 构建模型输入输出 (用标准化的数据)
    # 合并标准化 或 归一化 后的特征和目标变量
    data_prepared = None
    if config.scaling_method == 'standardization': # 标准化后的特征和目标
        data_prepared = np.column_stack((X_standardized, y_standardized))
    elif config.scaling_method == 'normalization':# 归一化后的特征和目标
        data_prepared = np.column_stack((X_normalized, y_normalized))
    feature_dim_start_col = 0
    feature_dim_end_col = X.shape[1] - 1 # 从第1列到第倒数第二列都是feature
    target_dim_col = X.shape[1] # 最后一列是target列
    # dataset.shape[0]行数据=1825，seq_len=30 -> 能产生1825-30=1795个window
    # feature列数：feature_dim_end_col-feature_dim_start_col+1个feature列=24（代码中遍历为feature_dim_start_col:feature_dim_end_col+1）展为1维向量：seq_len*feature_size = 30*24 = 720
    # X_seq: 窗口数*窗口大小*特征列数 = 1801 * (24 * 24) = 1801 * 576（展为1维） y_seq: 1801 * 1（单目标预测）  1801 * n(多目标预测)
    X_seq, y_seq = windows_select_single_label(data_prepared, feature_dim_start_col, feature_dim_end_col, config.seq_len, target_dim_col)
    print(f"在输入时间步长为:{config.seq_len}的前提下，数据集形状：{data_prepared.shape}，可以有:{X_seq.shape[0]}个batch")
    print(f"特征数据形状: {X_seq.shape},输入时间步:{config.seq_len} * 特征数:{config.enc_in} = {config.seq_len * config.enc_in}")
    if len(y_seq.shape) == 1:
        print(f"目标数据形状: {y_seq.shape}"+f"一共{y_seq.shape[0]}个batch，每个batch对应 1 列预测值")
    else:
        print(f"目标数据形状: {y_seq.shape}" + f"一共{y_seq.shape[0]}个batch，每个batch对应 {y_seq.shape[1]} 列预测值")

    X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
    if platform.system() == 'Linux with not enough RAM': # Linux服务器调不了虚拟内存，只能尝试通过内存映射来把数据集读到内存中，再进行数据集划分
        # 使用内存映射（memory-mapped files）来处理大型数据集
        # 将 X_seq 和 y_seq 保存到 .npy 文件
        np.save('/root/autodl-tmp/X_seq.npy', X_seq) # /root/autodl-tmp路径是我们autodl买的的数据盘，免费50GB, 我扩容到150GB了
        np.save('/root/autodl-tmp/y_seq.npy', y_seq)

        # 使用内存映射加载 .npy 文件
        X_seq_mmap = np.load('/root/autodl-tmp/X_seq.npy', mmap_mode='r')
        y_seq_mmap = np.load('/root/autodl-tmp/y_seq.npy', mmap_mode='r')

        # 数据集划分 80% 训练集，10% 验证集，10% 测试集, 确保数据集的顺序是随机的，X_seq 和 y_seq 都是 ndarray，切分数据时可能会导致内存占用过多，因为 train_test_split 会复制数组
        X_train, X_temp, y_train, y_temp = train_test_split(X_seq_mmap, y_seq_mmap, test_size=0.2, random_state=42,shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
    elif platform.system() == 'Windows' or 'Linux': # Windows可以通过调虚拟内存，来扩大内存，但Linux不行
        X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42,shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

    print(f"训练集大小: {X_train.shape}, {y_train.shape}")
    print(f"验证集大小: {X_val.shape}, {y_val.shape}")
    print(f"测试集大小: {X_test.shape}, {y_test.shape}")

    # 数据转换，numpy的ndarray 转为 torch.Tensor 类型对象, 并移动到 GPU 或 CPU
    X_train_tensor = torch.tensor(X_train, dtype=torch.float16).to(device) # 数据精度调小 float32->float16
    y_train_tensor = torch.tensor(y_train, dtype=torch.float16).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float16).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float16).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float16).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float16).to(device)

    # 创建训练集/验证集/测试集 数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # 创建数据集加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 定义选取模型，并移动到 GPU 或 CPU
    model_dict = ModelDict().get_model_dict()
    model = model_dict.get(config.model)(config).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss() # 对于回归任务，用MSE损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # 学习率设置放在ModelConfig中

    # 设置日志记录器
    log_dir = os.path.join(os.path.dirname(os.getcwd()),'log')
    print('日志目录：' + log_dir)
    model_name = type(model).__name__

    logger, logger_filename, logger_filepath = setup_logger(log_dir, model_name, config)
    logger.info('日志目录：' + log_dir)

    # 实例化 EarlyStopping
    early_stopping = EarlyStopping(logger=logger, patience=config.es_patience, verbose=config.es_verbose,
                                   delta=config.es_delta, path=config.es_path)

    # 查看配置
    print("模型训练前查看配置：")
    logger.info(f"\n本次训练配置config:{config}")
    print(config)

    # 初始化 GradScaler（训练时，使用混合精度）
    scaler = GradScaler()

    # 训练模型（训练时，使用混合精度）
    train_start_time = time.time()
    end_epoch = 0 # 记录在哪一次epoch时结束了train（earlyStopping）
    for epoch in tqdm(range(config.num_epochs), desc='Epochs'):
        end_epoch += 1
        model.train() # 设置模型为训练模式（启用Dropout和BatchNorm的训练行为）
        train_loss = 0.0  # 初始化训练损失
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad() # 清空上一步的梯度信息
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到设备CPU 或 GPU
            # 将输入数据调整为 (batch_size, seq_len, num_features) 形状以适应模型
            # inputs.size(0): 获取 inputs 张量的第一个维度大小，这通常是批次大小（batch_size）。
            # config.seq_len: 表示时间序列的长度（即输入序列的长度）。
            # -1: 是一个自动计算维度的占位符。PyTorch 会根据张量的总元素数和指定的其他维度来推断该维度的大小。
            # view 方法用于重塑张量的形状。它不改变数据本身，而是改变数据的表示方式。
            # 假设 inputs 原始形状为 (batch_size, some_length, some_channels)，通过 view 操作，它将被重塑为 (batch_size, config.seq_len, new_feature_size)，其中 new_feature_size 是根据原始形状自动推导出来的
            with autocast('cuda'):  # 使用 autocast 进行混合精度训练
                inputs = inputs.view(inputs.size(0), config.seq_len, -1) # (batch_size, seq_len*feature_size)=(32,720) -> (batch_size, seq_len, feature_size)=(32,30,24)
                outputs = model(inputs) # 将处理后的 inputs 传递给模型 model，进行前向传播
                if config.pred_len == 1: # 预测长度不同->outputs形状不同->给模型损失函数之前的处理方式就不同
                    # 输出方式1：使用每个batch，最后一个Channel的输出
                    # outputs = outputs.squeeze(dim=1)  # [batch_size, 1, channel] 调整为 [batch_size, channel] Channel就是feature_size
                    # outputs = outputs[:, -1]  # 对每一个batch, 仅使用最后一个Channel的输出，形状由 (batch_size, Channel) 变为 (batch_size, 1) 1为最后一个Channel的输

                    # 输出方式2：使用每个batch中，所有channel输出的平均值
                    # squeeze 函数的作用是移除张量（tensor）中指定维度（dim）大小为1的维度。如果不指定 dim 参数，它会移除所有大小为1的维度(注意维度从 0 开始计数)
                    # 如果张量中没有维度为 1 的维度，那么 squeeze 函数对该张量不会产生任何效果。具体来说，squeeze 只会尝试移除那些大小为 1 的维度，如果没有这样的维度，张量的形状将保持不变。
                    outputs = outputs.squeeze(dim=1) # 将形状从 [batch_size, 1, channel] 调整为 [batch_size, channel]
                    # mean(dim=1) 在第 1 个维度（即 channel 维度）上计算平均值，结果是对每个样本（对应 batch_size）得到一个标量（即单个值）
                    outputs = outputs.mean(dim=1) # 计算每个batch所有通道的平均值，形状变化：[batch_size, channel] -> [batch_size, 1] 1为相应批次中所有channel的平均值，每个批次的预测结果是基于所有通道的平均值
                else: # pred_len > 1
                    # 输出方式1：选择 pred_len 维度上最后一个时间步长的输出，然后再对 channel 维度进行平均
                    # outputs = outputs[:, -1, :].mean(dim=1)  # 形状变为 [32]
                    # 输出方式2: 对 pred_len 和 channel 维度进行平均
                    outputs = outputs.mean(dim=[1, 2])
                loss = criterion(outputs.squeeze(), targets) # 计算当前批次的损失 targets维度为[32]
            scaler.scale(loss).backward() # 使用 scaler 缩放损失并进行反向传播计算梯度，backward() 会自动计算出损失函数相对于所有可训练参数的梯度，并将这些梯度存储在每个参数的 grad 属性中
            # 使用 scaler 进行模型参数梯度更新
            scaler.step(optimizer)
            # 更新 scaler 状态
            scaler.update()
            train_loss += loss.item() # 累加当前批次的损失
        train_loss /= len(train_loader)

        # 设置模型为评估模式（禁用Dropout并使用BatchNorm的运行均值和方差）
        model.eval()
        val_loss = 0.0 # 初始化验证损失
        # 禁用梯度计算，提高评估效率
        with torch.no_grad():
            with autocast('cuda'):  # 使用 autocast 进行验证时的前向传播
            # 遍历验证数据集的每一个批次
                for inputs, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", leave=False):# 遍历验证数据集的每一个批次
                    inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到设备CPU 或 GPU
                    # 将输入数据调整为 (batch_size, seq_len, num_features) 形状以适应模型
                    outputs = model(inputs.view(inputs.size(0), config.seq_len, -1))
                    if config.pred_len == 1:
                        # 输出方式2：使用每个batch中，所有channel输出的平均值
                        outputs = outputs.squeeze(dim=1)  # 将形状从 [batch_size, 1, channel] 调整为 [batch_size, channel]。
                        outputs = outputs.mean(dim=1)
                    else:
                        # 输出方式1：选择 pred_len 维度上最后一个时间步长的输出，然后再对 channel 维度进行平均
                        # outputs = outputs[:, -1, :].mean(dim=1)  # 形状变为 [32]
                        # 输出方式2: 对 pred_len 和 channel 维度进行平均
                        outputs = outputs.mean(dim=[1, 2])
                    loss = criterion(outputs.squeeze(), targets) # 计算当前批次的损失
                    val_loss += loss.item() # 累加当前批次的损失
        val_loss /= len(val_loader)

        # 记录训练和验证损失到日志文件
        logger.info(f'Epoch {epoch + 1}/{config.num_epochs}, Train Loss(MSE): {train_loss}, Validation Loss(MSE): {val_loss}')
        print(f'Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

        # 检查是否应触发早停
        early_stopping(val_loss, model)

        # 如果早停条件满足，退出训练
        if early_stopping.early_stop:
            logger.info("Early stopping")
            print("Early stopping")
            break
    train_end_time = time.time()
    train_cost_time = train_end_time - train_start_time
    logger.info(f"训练 {end_epoch} 次epoch，训练时长: {train_cost_time} 秒")
    print(f"训练 {end_epoch} 次epoch，训练时长: {train_cost_time} 秒")

    # 在训练结束后加载最优的模型参数
    model.load_state_dict(torch.load('current_best_checkpoint.pt'))


    # 保存模型状态字典(模型参数)到 model_save_path (checkpoint目录下)
    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    state_dict_dir = os.path.join(os.path.dirname(os.getcwd()), 'checkpoint')
    if not os.path.exists(state_dict_dir):
        os.makedirs(state_dict_dir)
    print('模型参数保存目录：' + state_dict_dir)
    model_save_path = os.path.join(state_dict_dir,
                                   f"{model_name}_{current_time}_bs{config.batch_size}_seq{config.seq_len}" +
                                   f"_enc{config.enc_in}_pred_len{config.pred_len}_individual{config.individual}_checkpoint.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f'Model saved to {model_save_path}')
    print(f'Model saved to {model_save_path}')


    # 评估模型(注意模型评估时的数据精度，因为模型训练用的是混合精度训练)
    model.eval() # 在评估过程中不更新模型的状态字典（超参数），评估后保存的模型参数和评估前保存的参数是完全相同的
    test_mse_loss = 0.0 # MSE 累计
    test_mae_loss = 0.0 # MAE 累计
    with torch.no_grad():
        for inputs, targets in test_loader:
            # 模型评估时，将数据精度都调为float32
            inputs, targets = inputs.to(device).to(torch.float32), targets.to(device).to(torch.float32)  # 将数据移动到设备CPU 或 GPU
            outputs = model(inputs.view(inputs.size(0), config.seq_len, -1))
            if config.pred_len == 1:
                # 输出方式2：使用每个batch中，所有channel输出的平均值
                outputs = outputs.squeeze(dim=1)  # 将形状从 [batch_size, 1, channel] 调整为 [batch_size, channel]。
                outputs = outputs.mean(dim=1)
            else:
                # 输出方式1：选择 pred_len 维度上最后一个时间步长的输出，然后再对 channel 维度进行平均
                # outputs = outputs[:, -1, :].mean(dim=1)  # 形状变为 [32]
                # 输出方式2: 对 pred_len 和 channel 维度进行平均
                outputs = outputs.mean(dim=[1, 2])
            # 计算 MSE 损失
            loss = criterion(outputs.squeeze(), targets)
            test_mse_loss += loss.item()
            # 计算 MAE 损失 (使用 sklearn)
            test_mae_loss += mean_absolute_error(targets.cpu().numpy(), outputs.squeeze().cpu().numpy())
    # 计算平均 MSE 和 MAE
    test_mse_loss /= len(test_loader)
    test_mae_loss /= len(test_loader)

    # 记录测试损失到日志文件
    logger.info(f'Test Loss ({config.scaling_method} Scale) MSE: {test_mse_loss}')
    print(f'Test Loss ({config.scaling_method} Scale) MSE: {test_mse_loss}')
    logger.info(f'Test Loss ({config.scaling_method} Scale) MAE: {test_mae_loss}')
    print(f'Test Loss ({config.scaling_method} Scale) MAE: {test_mae_loss}')


    # 反标准化目标变量预测值, 在测试集上评估模型的性能
    model.eval()  # 设置模型为评估模式
    predictions = [] # 装整个测试集的所有预测结果
    with torch.no_grad():
        for inputs, _ in test_loader:
            # 模型评估时，将数据精度都调为float32
            inputs = inputs.to(device).to(torch.float32)  # 将数据移动到设备CPU 或 GPU
            # 输入数据调整为 (batch_size, sequence_length, num_features) 形状以适应模型
            inputs = inputs.view(inputs.size(0), config.seq_len, -1)
            outputs = model(inputs) # 生成预测值：将输入数据传入模型进行前向传播，得到预测结果
            if config.pred_len == 1:
                # 输出方式2：使用每个batch中，所有channel输出的平均值
                outputs = outputs.squeeze(dim=1)  # 将形状从 [batch_size, 1, channel] 调整为 [batch_size, channel]。
                outputs = outputs.mean(dim=1) # 平均所有通道的值 形状：[batch_size]
            else:
                # 输出方式1：选择 pred_len 维度上最后一个时间步长的输出，然后再对 channel 维度进行平均
                # outputs = outputs[:, -1, :].mean(dim=1)  # 形状变为 [32]
                # 输出方式2: 对 pred_len 和 channel 维度进行平均
                outputs = outputs.mean(dim=[1, 2]) # 时间步pred_len和所有通道channel的平均值[32]
            predictions.extend(outputs.squeeze().cpu().numpy()) # 将每个批次的预测结果将 PyTorch 张量 转换为 NumPy 数组并添加到 predictions 列表中; 只有当张量在 CPU 上时，才能调用 numpy() 方法，因此先调用 cpu()
            # print(len(predictions)) prediction最终大小为一维数组，长度为测试集batch_size, 对应测试集每个batch的预测值

    # 反标准化/反归一化预测值
    predictions = np.array(predictions)
    # 变为一个二维数组，其中每一行是一个预测值 [1,180] -> [180,1]
    if config.scaling_method == 'standardization':
        predictions_original_scale = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten() # 对预测值进行反标准化，将其从标准化后的尺度变换回原始尺度
    elif config.scaling_method == 'normalization':
        predictions_original_scale = scaler_y_minmax.inverse_transform(predictions.reshape(-1, 1)).flatten() # 对预测值进行反归一化，将其从归一化后的尺度变换回原始尺度

    # 反标准化/反归一化真实值 [180,1]
    if config.scaling_method == 'standardization':
        y_test_original_scale = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten() # 对真实值进行反标准化
    elif config.scaling_method == 'normalization':
        y_test_original_scale = scaler_y_minmax.inverse_transform(y_test.reshape(-1, 1)).flatten() # 对真实值进行反归一化

    # 保存测试集上的反标准化/反归一化的真实值（Original Scale）预测值和真实值
    df_test_predict_vs_true_results = pd.DataFrame({
        'Predictions': predictions_original_scale,
        'True Values': y_test_original_scale
    })
    predict_data_dir = os.path.join(os.path.dirname(os.getcwd()),'predict_data')
    if not os.path.exists(predict_data_dir):
        os.makedirs(predict_data_dir)
    # 保存到 CSV 文件
    output_csv_path = os.path.join(predict_data_dir, f'predictions_vs_true_values(Original Scale)-{model_name}-{dataset_name}-{current_time}-sl{config.seq_len}_{config.scaling_method}.csv')
    df_test_predict_vs_true_results.to_csv(output_csv_path, index=False)
    logger.info(f"测试集预测值和真实值数据存储：Predictions and True Values saved to {output_csv_path}")
    print(f"测试集预测值和真实值数据存储：Predictions and True Values saved to {output_csv_path}")

    # 计算反标准化后的 预测值和真实值之间的均方误差MSE
    test_mse_loss_original_scale = mean_squared_error(y_test_original_scale, predictions_original_scale)
    logger.info(f'Test Loss (Original Scale) MSE: {test_mse_loss_original_scale}')
    print(f'Test Loss (Original Scale) MSE: {test_mse_loss_original_scale}') # 对较大的预测误差给予更大的惩罚
    test_mae_loss_original_scale = mean_absolute_error(y_test_original_scale, predictions_original_scale)
    logger.info(f'Test Loss (Original Scale) MAE: {test_mae_loss_original_scale}\n')
    print(f'Test Loss (Original Scale) MAE: {test_mae_loss_original_scale}\n') # 对异常值不非常敏感的度量

    # 绘制预测值和真实值的对比图
    predict_pic_dir = os.path.join(os.path.dirname(os.getcwd()),'predict_pic')
    if not os.path.exists(predict_pic_dir):
        os.makedirs(predict_pic_dir)
    plt.figure(figsize=(12, 6)) # 创建一个新的图形对象，大小为 12x6 英寸
    plt.plot(y_test_original_scale, label='True Values', color='b')
    plt.plot(predictions_original_scale, label='Predictions', color='r')
    plt.xlabel('Sample Index(Day)')
    plt.ylabel('OT')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(predict_pic_dir, f'predictions_vs_true_values-{model_name}-{dataset_name}-{current_time}-sl{config.seq_len}_{config.scaling_method}.png')
    plt.savefig(plot_path)
    # plt.show()
    print(f"\n================实验结束：seq_len = {config.seq_len}, 模型超参数和测试集预测结果已保存。==============\n")
