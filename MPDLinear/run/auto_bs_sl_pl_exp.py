'''
对比实验：batch_size, seq_len, pred_len的选择（BKC获取的一部分, 实验模型MPDLinear_SOTA
固定其他参数，跑消融实验，在某一数据集上，选择对模型预测最好的 batch_size, seq_len, pred_len
'''

import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from MPDLinear.model.MPDLinear_SOTA import MPDLinear_SOTA
from MPDLinear.data_process.data_processor import load_data, clean_weather_and_save, divide_data_by_geographic, feature_engineering, windows_select_single_label
from MPDLinear.config.ModelConfig import ModelConfig
from util.EarlyStopping import EarlyStopping
from util.logger import setup_logger
from datetime import datetime
import os
import matplotlib.pyplot as plt

# 定义自动化实验函数
def run_experiment(config, scaler_X, X, scaler_y ,y, batch_size, seq_len, pred_len):
    '''
    此函数负责运行一次实验，包括配置模型、训练、验证和测试。它接受多个参数来灵活地运行不同配置的实验：
    :param config: 模型配置对象
    :param X: 标准化后的特征和
    :param y: 标准化后的目标变量数据
    :param batch_size: 运行实验时的批次大小
    :param seq_len: 序列长度
    :param pred_len: 预测长度
    :return:
    '''
    # 更新配置
    config.batch_size = batch_size
    config.seq_len = seq_len
    config.pred_len = pred_len

    # bs_sl_pl实验中，模型配置的固定参数
    config.individual = True
    config.enc_in = X.shape[1] # 特征数
    config.num_epochs = 100
    # EarlyStopping模块配置
    config.es_patience = 5
    config.es_verbose = True
    config.es_delta = 0.00001
    config.es_path = 'current_best_checkpoint.pt'
    config.decomposition_kernel_size = 25

    # 数据准备: 构建模型输入输出 (用标准化的数据)
    data_prepared = np.column_stack((X, y))
    feature_dim_start_col = 0
    feature_dim_end_col = X.shape[1] - 1
    target_dim_col = X.shape[1]
    X_seq, y_seq = windows_select_single_label(data_prepared, feature_dim_start_col, feature_dim_end_col, config.seq_len, target_dim_col)

    # 数据划分：训练集:测试集:验证集 = 8:1:1
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 数据转换，numpy的ndarray 转为 torch.Tensor 类型对象
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 创建训练集/验证集/测试集 数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # 创建数据集加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 模型定义与损失函数、优化器初始化
    model = MPDLinear_SOTA(config)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 日志记录
    model_name = type(model).__name__
    log_dir = os.path.join(os.path.dirname(os.getcwd()), 'bs_sl_pl_exp_log')
    logger, _, _ = setup_logger(log_dir, model_name, config)
    # 早停机制
    early_stopping = EarlyStopping(logger=logger, patience=config.es_patience, verbose=config.es_verbose,
                                   delta=config.es_delta, path=config.es_path)

    # 模型训练
    train_start_time = time.time()
    end_epoch = 0  # 记录在哪一次epoch时结束了train（earlyStopping）
    for epoch in range(config.num_epochs):
        end_epoch += 1
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs = inputs.view(inputs.size(0), config.seq_len, -1)
            outputs = model(inputs)
            if config.pred_len == 1:
                outputs = outputs.squeeze(dim=1).mean(dim=1)
            else:
                outputs = outputs.mean(dim=[1, 2])
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 跑验证集
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.view(inputs.size(0), config.seq_len, -1)
                outputs = model(inputs)
                if config.pred_len == 1:
                    outputs = outputs.squeeze(dim=1).mean(dim=1)
                else:
                    outputs = outputs.mean(dim=[1, 2])
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        logger.info(f'Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

        # 检查是否应触发早停
        early_stopping(val_loss, model)

        # 如果早停条件满足，退出训练
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    train_end_time = time.time()
    train_cost_time = train_end_time - train_start_time
    logger.info(f"训练 {end_epoch} 次epoch，训练时长: {train_cost_time} 秒")

    # 模型测试 评估模型
    model.load_state_dict(torch.load(config.es_path))
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.view(inputs.size(0), config.seq_len, -1)
            outputs = model(inputs)
            if config.pred_len == 1:
                outputs = outputs.squeeze(dim=1).mean(dim=1)
            else:
                outputs = outputs.mean(dim=[1, 2])
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item()
    test_loss /= len(test_loader)

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

    # 反标准化目标变量预测值, 在测试集上评估模型的性能，计算在在Original Scale的均方误差MSE
    model.eval()  # 设置模型为评估模式
    predictions = []  # 装整个测试集的所有预测结果
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.view(inputs.size(0), config.seq_len, -1)
            outputs = model(inputs)  # 生成预测值：将输入数据传入模型进行前向传播，得到预测结果
            if config.pred_len == 1:
                # 输出方式2：使用每个batch中，所有channel输出的平均值
                outputs = outputs.squeeze(dim=1)  # 将形状从 [batch_size, 1, channel] 调整为 [batch_size, channel]。
                outputs = outputs.mean(dim=1)  # 平均所有通道的值 形状：[batch_size]
            else:
                outputs = outputs.mean(dim=[1, 2])  # 时间步pred_len和所有通道channel的平均值[32]
            predictions.extend(outputs.squeeze().cpu().numpy())  # 将每个批次的预测结果将 PyTorch 张量 转换为 NumPy 数组并添加到 predictions 列表中; 只有当张量在 CPU 上时，才能调用 numpy() 方法，因此先调用 cpu()
    # 反标准化预测值
    predictions = np.array(predictions)
    # 变为一个二维数组，其中每一行是一个预测值 [1,180] -> [180,1]
    predictions_original_scale = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()  # 对预测值进行反标准化，将其从标准化后的尺度变换回原始尺度

    # 反标准化真实值 [180,1]
    y_test_original_scale = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()  # 对真实值进行反标准化

    # 计算反标准化后的 预测值和真实值之间的均方误差MSE
    test_mse_loss_original_scale = mean_squared_error(y_test_original_scale, predictions_original_scale)
    logger.info(f'Test Loss (Original Scale) MSE: {test_mse_loss_original_scale}')
    print(f'Test Loss (Original Scale) MSE: {test_mse_loss_original_scale}')  # 对较大的预测误差给予更大的惩罚
    test_mae_loss_original_scale = mean_absolute_error(y_test_original_scale, predictions_original_scale)
    logger.info(f'Test Loss (Original Scale) MAE: {test_mae_loss_original_scale}')
    print(f'Test Loss (Original Scale) MAE: {test_mae_loss_original_scale}')  # 对异常值不非常敏感的度量


    return train_cost_time, val_loss, test_loss, test_mse_loss_original_scale, test_mae_loss_original_scale

# 主函数
def main():
    # 初始化模型配置类对象
    config = ModelConfig()
    dataset_path = '../datasets/weather/weather1_by_day/weather_2013_2017_china.csv'

    # 加载数据 + 数据清洗 + 按地域划分数据集(beijing/shanghai/chongqing) + 特征工程(为每个city avg或expand county列)
    ori_dataset = load_data(dataset_path)
    cleaned_dataset = clean_weather_and_save(ori_dataset)
    beijing_data, shanghai_data, chongqing_data = divide_data_by_geographic()
    avg_data_list, feature_expand_data_list = feature_engineering([beijing_data])

    # 选用avg county的beijing city数据
    avg_data_beijing = avg_data_list[0]

    # 插值法+前向+后向填充缺失值
    avg_data_beijing.interpolate(method='linear', inplace=True)
    avg_data_beijing.fillna(method='ffill', inplace=True)
    avg_data_beijing.fillna(method='bfill', inplace=True)

    # 特征选择
    selected_features = ['pressure', 'wind_speed', 'wind_direction', 'humidity', 'rain20', 'rain08', 'cloud', 'visibility',
                         'sunny', 'cloudy', 'rain', 'fog', 'haze', 'dust', 'thunder', 'lightning', 'snow', 'hail', 'wind',
                         'year', 'month', 'week', 'quarter', 'day']

    # 提取特征feature和目标变量target
    X = avg_data_beijing[selected_features]
    y = avg_data_beijing['temperature'] # 单特征预测

    # 标准化feature和target
    scaler_X = StandardScaler()
    X_standardized = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_standardized = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    # 模型参数配置种 要进行实验的batch_size/seq_len/pred_len参数，其他参数写死在run_experiment()中
    batch_sizes = [16, 32, 64, 128]
    seq_lens = [7, 30, 90, 180, 365]
    pred_lens = [1, 3, 5, 7, 10, 30]

    results = []

    # 对不同 batch_size/seq_len/pred_len 进行实验
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for pred_len in pred_lens:
                print(f'\nRunning experiment with batch_size={batch_size}, seq_len={seq_len}, pred_len={pred_len}')
                train_time, val_loss, test_loss, test_mse_loss_original_scale, test_mae_loss_original_scale = run_experiment(
                    config, scaler_X, X_standardized, scaler_y, y_standardized, batch_size, seq_len, pred_len)
                results.append([batch_size, seq_len, pred_len, train_time, val_loss, test_loss, test_mse_loss_original_scale, test_mae_loss_original_scale])
                print(f'Completed: train_time={train_time} s, test_loss_original_scale(MSE)={test_mse_loss_original_scale}'+
                      f', test_loss_original_scale(MAE)={test_mae_loss_original_scale}\n')

    df = pd.DataFrame(results, columns=['batch_size', 'seq_len', 'pred_len', 'train_time (s)', 'val_loss', 'test_loss', 'test_loss_original_scale(MSE)', 'test_loss_original_scale(MAE)'])
    exp_data_file = 'bs_sl_pl_experiment_results.csv'
    df.to_csv(exp_data_file, index=False)
    print(f'All experiments completed. Results saved to {exp_data_file}.')

if __name__ == '__main__':
    main()
