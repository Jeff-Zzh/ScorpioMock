import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from MPDLinear.config.ModelConfig import ModelConfig
from MPDLinear.data_process.data_processor import windows_select_single_label, load_data, \
    feature_engineering_for_electricity_and_save
from MPDLinear.model.model_dict import ModelDict
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 加载数据
dataset_path = '../datasets/9_LTSF_dataset/ori/electricity.csv'
ori_dataset = load_data(dataset_path)

expand_time_feature_dataset = feature_engineering_for_electricity_and_save(ori_dataset)

# 选择数据集
dataset_name = 'electricity'
dataset_exp = expand_time_feature_dataset

# 特征选择/Label选择
selected_features = dataset_exp.columns[:328]
X = dataset_exp[selected_features]  # 特征变量
y = dataset_exp['OT']  # 目标变量

# 标准化特征
scaler_X = StandardScaler()
X_standardized = scaler_X.fit_transform(X)
# 标准化目标变量
scaler_y = StandardScaler()
y_standardized = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# 归一化特征
scaler_X_minmax = MinMaxScaler()
X_normalized = scaler_X_minmax.fit_transform(X)
# 归一化目标变量
scaler_y_minmax = MinMaxScaler()
y_normalized = scaler_y_minmax.fit_transform(y.values.reshape(-1, 1)).flatten()

# 配置
config = ModelConfig() # 初始化模型参数配置
config.model = 'MPDLinear_SOTA'
config.batch_size = 4 # 数据集有1825个样本。通常，对于小型数据集，较小的 batch_size 会更合适，如 16、32(cpu) 或 64,128(gpu)。这样可以更充分地利用数据并防止内存溢出
# seq_len选择
# 常用起点: 可以从 30 天（一个月的时间序列数据）开始。这通常是一个合理的起点，既能捕捉短期趋势，又不会过长。
# 根据模型调整: 如果在训练过程中发现模型表现良好，可以逐步增加 seq_len，例如增加到 60 天、90 天或 180 天，看看是否能进一步提升模型性能
config.seq_len = None # {24，48，72，96，120，144，168，192，336，504，672，720}
# pred_len选择
# 推荐起点: 开始尝试 pred_len = 7（一周）或 pred_len = 10（十天），这些是常用的时间段，并且容易观察预测的准确性。
# 调整方向: 如果模型训练效果较好，你可以逐步增加 pred_len，如 14 天或 30 天，直到找到一个适合的平衡点。
config.pred_len = 5 # 1,3,5,7,10,30 数据集有 1825 个时间步，代表了按天采样5年的时间跨度。对于如此长时间的数据，如果 pred_len 设置得太大，模型可能难以学习到有效的长期依赖关系，因此需要谨慎选择。
config.individual = True
config.enc_in = len(selected_features) # electricity数据集特征数
config.num_epochs = 100
# EarlyStopping模块配置
config.es_patience = 5 # 100 跑全epoch(晚上跑)
config.es_verbose = True
config.es_delta = 0.00001
config.es_path = 'current_best_checkpoint.pt'
config.decomposition_kernel_size = 25
config.learning_rate = 0.0001 # 0.001 -> 0.0001 -> 0.00001
config.scaling_method = 'standardization' # 选择缩放方法 标准化/归一化
config.device = 'gpu'

device = None # torch所用设备
if config.device == 'gpu':
# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-----Using device: {device}-----")
else:
    device = torch.device('cpu')
    print(f"-----Using device: {device}-----")

config.seq_len = 504 # 设置当前的 seq_len
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


# 数据集划分 80% 训练集，10% 验证集，10% 测试集, 确保数据集的顺序是随机的
X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=True)
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
criterion = nn.MSELoss()  # 对于回归任务，用MSE损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # 学习率设置放在ModelConfig中

# 加载保存的最佳模型
model_name = type(model).__name__
model_path = 'current_best_checkpoint.pt'
model.load_state_dict(torch.load(model_path,weights_only=True))
model.eval()

# 评估模型
model.eval()
test_mse_loss = 0.0  # MSE 累计
test_mae_loss = 0.0  # MAE 累计
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到设备CPU 或 GPU
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
# logger.info(f'Test Loss ({config.scaling_method} Scale) MSE: {test_mse_loss}')
print(f'Test Loss ({config.scaling_method} Scale) MSE: {test_mse_loss}')
# logger.info(f'Test Loss ({config.scaling_method} Scale) MAE: {test_mae_loss}')
print(f'Test Loss ({config.scaling_method} Scale) MAE: {test_mae_loss}')

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
# logger.info(f'Model saved to {model_save_path}')
print(f'Model saved to {model_save_path}')

# 反标准化目标变量预测值, 在测试集上评估模型的性能
model.eval()  # 设置模型为评估模式
predictions = []  # 装整个测试集的所有预测结果
with torch.no_grad():
    for inputs, _ in test_loader:
        # 输入数据调整为 (batch_size, sequence_length, num_features) 形状以适应模型
        inputs = inputs.to(device)  # 将数据移动到设备CPU 或 GPU
        inputs = inputs.view(inputs.size(0), config.seq_len, -1)
        outputs = model(inputs)  # 生成预测值：将输入数据传入模型进行前向传播，得到预测结果
        if config.pred_len == 1:
            # 输出方式2：使用每个batch中，所有channel输出的平均值
            outputs = outputs.squeeze(dim=1)  # 将形状从 [batch_size, 1, channel] 调整为 [batch_size, channel]。
            outputs = outputs.mean(dim=1)  # 平均所有通道的值 形状：[batch_size]
        else:
            # 输出方式1：选择 pred_len 维度上最后一个时间步长的输出，然后再对 channel 维度进行平均
            # outputs = outputs[:, -1, :].mean(dim=1)  # 形状变为 [32]
            # 输出方式2: 对 pred_len 和 channel 维度进行平均
            outputs = outputs.mean(dim=[1, 2])  # 时间步pred_len和所有通道channel的平均值[32]
        predictions.extend(
            outputs.squeeze().cpu().numpy())  # 将每个批次的预测结果将 PyTorch 张量 转换为 NumPy 数组并添加到 predictions 列表中; 只有当张量在 CPU 上时，才能调用 numpy() 方法，因此先调用 cpu()
        # print(len(predictions)) prediction最终大小为一维数组，长度为测试集batch_size, 对应测试集每个batch的预测值

# 反标准化/反归一化预测值
predictions = np.array(predictions)
# 变为一个二维数组，其中每一行是一个预测值 [1,180] -> [180,1]
if config.scaling_method == 'standardization':
    predictions_original_scale = scaler_y.inverse_transform(
        predictions.reshape(-1, 1)).flatten()  # 对预测值进行反标准化，将其从标准化后的尺度变换回原始尺度
elif config.scaling_method == 'normalization':
    predictions_original_scale = scaler_y_minmax.inverse_transform(
        predictions.reshape(-1, 1)).flatten()  # 对预测值进行反归一化，将其从归一化后的尺度变换回原始尺度

# 反标准化/反归一化真实值 [180,1]
if config.scaling_method == 'standardization':
    y_test_original_scale = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()  # 对真实值进行反标准化
elif config.scaling_method == 'normalization':
    y_test_original_scale = scaler_y_minmax.inverse_transform(y_test.reshape(-1, 1)).flatten()  # 对真实值进行反归一化

# 保存测试集上的反标准化/反归一化的真实值（Original Scale）预测值和真实值
df_test_predict_vs_true_results = pd.DataFrame({
    'Predictions': predictions_original_scale,
    'True Values': y_test_original_scale
})
predict_data_dir = os.path.join(os.path.dirname(os.getcwd()), 'predict_data')
if not os.path.exists(predict_data_dir):
    os.makedirs(predict_data_dir)
# 保存到 CSV 文件
output_csv_path = os.path.join(predict_data_dir,
                               f'predictions_vs_true_values(Original Scale)-{model_name}-{dataset_name}-{current_time}-sl{config.seq_len}.csv')
df_test_predict_vs_true_results.to_csv(output_csv_path, index=False)
# logger.info(f"测试集预测值和真实值数据存储：Predictions and True Values saved to {output_csv_path}")
print(f"测试集预测值和真实值数据存储：Predictions and True Values saved to {output_csv_path}")

# 计算反标准化后的 预测值和真实值之间的均方误差MSE
test_mse_loss_original_scale = mean_squared_error(y_test_original_scale, predictions_original_scale)
# logger.info(f'Test Loss (Original Scale) MSE: {test_mse_loss_original_scale}')
print(f'Test Loss (Original Scale) MSE: {test_mse_loss_original_scale}')  # 对较大的预测误差给予更大的惩罚
test_mae_loss_original_scale = mean_absolute_error(y_test_original_scale, predictions_original_scale)
# logger.info(f'Test Loss (Original Scale) MAE: {test_mae_loss_original_scale}\n')
print(f'Test Loss (Original Scale) MAE: {test_mae_loss_original_scale}\n')  # 对异常值不非常敏感的度量

# 绘制预测值和真实值的对比图
predict_pic_dir = os.path.join(os.path.dirname(os.getcwd()), 'predict_pic')
if not os.path.exists(predict_pic_dir):
    os.makedirs(predict_pic_dir)
plt.figure(figsize=(12, 6))  # 创建一个新的图形对象，大小为 12x6 英寸
plt.plot(y_test_original_scale, label='True Values', color='b')
plt.plot(predictions_original_scale, label='Predictions', color='r')
plt.xlabel('Sample Index(Day)')
plt.ylabel('Temperature')
plt.title('Predictions vs True Values')
plt.legend()
plt.grid(True)
plot_path = os.path.join(predict_pic_dir, f'predictions_vs_true_values_{current_time}_{model_name}.png')
plt.savefig(plot_path)
plt.show()
print(f"\n================实验结束：seq_len = {config.seq_len}, 模型超参数和测试集预测结果已保存。==============\n")