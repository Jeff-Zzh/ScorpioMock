import pandas as pd
import numpy as np
import torch
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    '''
    PyTorch 数据集类
    将时间序列数据组织成适合模型训练的格式
    将序列长度（seq_len）的连续数据切片，并将这些切片作为输入，同时将后一个时间步的值作为目标
    '''

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]  # 切片 [0.1, 0.2, 0.3] 输入序列长度为3
        y = self.data[idx + self.seq_len]  # 数值 0.4 预测目标target是下一个时间步的值
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)  # 创建张量，指定张量类型


def clean_weather_and_save(weather):
    '''
    数据清洗
    :param weather: 数据集dataset
    :return:
    '''
    # print(weather["cloud"].describe())

    # 处理日期时间 将 weather 数据框中的 date 列转换为日期时间格式
    date = pd.to_datetime(weather['date'].apply(lambda x: str(x)))  # date 列中的每个值都被转换为 字符串 格式，然后再转换为 日期时间 格式

    # 提取时间属性
    weather['year'] = date.dt.year
    weather['month'] = date.dt.month
    # weather['week'] = date.dt.weekofyear
    weather['week'] = date.dt.isocalendar().week
    weather['quarter'] = date.dt.to_period('Q').astype('str')[:-2].apply(lambda x: x[-1]).astype('int')
    weather['day'] = date.dt.dayofyear

    # 将dataset中所有的无效数据替换成NAN
    # weather[weather == 999999] = np.NaN
    # weather[weather == 999990] = np.NaN
    weather.replace(999999, np.nan, inplace=True)
    weather.replace(999990, np.nan, inplace=True)

    # 先将 NaN 填充为一个特定值（例如 0）
    weather['wind_direction'].fillna(0, inplace=True)

    # 去掉 wind_direction 和 phenomenon 属性
    # weather = weather.drop(["wind_direction","phenomenon"],axis=1)

    # 修正错误数据 999013->13
    # 去掉 wind_direction 列中以 999 开头的前 3 位
    # 将所有值转换为整数
    weather['wind_direction'] = weather['wind_direction'].astype(float).astype(int)
    # 提取后三位并转换为整数
    weather['wind_direction'] = weather['wind_direction'].astype(str).str[-3:].astype(int)
    # 将 wind_direction 限制在 0-360 之间
    weather['wind_direction'] = weather['wind_direction'].apply(lambda x: x if 0 <= x <= 360 else None)

    # 删除 phenomenon 属性列，用one-hot编码代替（one-hot编码已经在dataset中存在）
    weather = weather.drop(["phenomenon"], axis=1)

    print("清洗后数据集：")
    print(weather.describe())
    # 保存清洗后的数据
    weather.to_csv('../datasets/weather/weather1_by_day/clean_weather_2013_2017_china.csv', index=False)
    return weather


def load_data(dataset_path):
    '''
    导入数据
    :param dataset_path:
    :return:
    '''
    return pd.read_csv(dataset_path)


def normalize(df:DataFrame, columns:list):
    # 归一化函数，将数据缩放到 [0, 1] 范围内
    result = df.copy()
    for column in columns:
        max_value = df[column].max()
        min_value = df[column].min()
        result[column] = (df[column] - min_value) / (max_value - min_value)
    return result


def preprocessing_data(city, dataset, selected_features, seq_len, batch_size=32):
    '''
    数据预处理
    :param city: 预处理的数据集（按城市分类）
    :param dataset: 要预处理的数据集 DataFrame
    :param selected_features:  时间序列预测的数值型特征 list
    :param seq_len: 序列长度，模型使用长度为 seq_len 的过去数据来预测下一个时间步或未来一段时间的值，每个样本seq_len个时间步
    :param batch_size: 批量大小 每次训练迭代中用于更新模型超参数的样本数，在训练过程中，将数据分成多个批次，每个批次包含 batch_size 个样本。
    这样可以在内存受限的情况下进行模型训练，并提高训练速度，如果 batch_size = 32，这意味着模型每次迭代会处理32个样本。
    :return:
    '''

    # 缺失值处理
    # 填充缺失值(将所有缺失值替换为它之前最近的一个非缺失值,将前向填充后仍然存在的缺失值替换为它之后最近的一个非缺失值)
    # 插值法利用数据的趋势和模式进行填补，而前向填充和后向填充则确保了所有缺失值都能被有效填补。这样处理后的数据将更加完整，有助于提高后续数据分析和建模的准确性
    dataset.interpolate(method='linear', inplace=True)
    dataset.fillna(method='ffill', inplace=True)
    dataset.fillna(method='bfill', inplace=True)

    # 特征选择，选择特征列
    dataset = dataset[selected_features]

    # 数据标准化 数据的均值为0，标准差为1 有助于加快模型训练过程并提高模型性能 特别是对于那些依赖梯度下降优化算法的模型

    dataset.to_csv(f'../datasets/weather/weather1_by_day/divide_by_city/standardized/{city}_standardized_weather.csv', index=False)
    # 数据归一化
    normalized_dataset = normalize(dataset, selected_features)
    normalized_dataset.to_csv(f'../datasets/weather/weather1_by_day/divide_by_city/normalized/{city}_normalized_weather.csv', index=False)

    # 特征生成
    # 数据集打乱
    # 数据集分割 将数据集划分为训练集、验证集和测试集
    # get_train_val_test_data(dataset, False, seq_len, ) # 单目标预测
    # get_train_val_test_data_multi() # 多目标预测

    # # 时间序列特征工程（如果需要）对于时间序列数据，可以考虑生成滞后特征、滚动统计特征等
    #
    #
    # # 将数据转换为 numpy 数组
    # dataset = dataset.values
    #
    # # 定义数据集和数据加载器
    # dataset = TimeSeriesDataset(dataset, seq_len)  # 数据集 数据 + 序列长度
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 每个批次加载的数据量为 32。即，每次从数据集中加载 32 个样本
    #
    # return dataloader
    return ''


'''====================================================单目标预测数据集生成============================================================'''
def windows_select_single_label(data_prepared, feature_dim_start_col, feature_dim_end_col, sequence_length, target_dim_col):
    '''
    从输入数据中选择固定长度的时间窗口（子序列），并为每个窗口生成相应的特征和标签。每个窗口包含指定数量的时间步（sequence_length），
    并将这些时间步的特征平铺成一个一维向量。
    在时间序列数据处理中，窗口是指一段连续的时间序列数据。通过窗口化操作，可以从时间序列中提取多个固定长度的子序列，每个子序列作为模型的一个输入
    :param data_prepared: 输入的数据集，通常是一个二维数组，每行是一个时间步的特征向量
    :param feature_dim_start_col: 选择的起始特征列 默认选择所有特征
    :param feature_dim_end_col：选择的结束特征列
    :param sequence_length:时间序列的长度（窗口大小）
    :param target_dim_col: 预测目标维度的列 （label的列）
    :return:
    '''
    features = []  # 所装元素为：每个batch对应的(seq_len * channels)，每个batch对应一个seq_len的特征列平铺，即每个batch对应的矩阵形状是：（seq_len * feature_size）
    labels = []
    for i in range(data_prepared.shape[0] - sequence_length):  # 可提取的窗口数量 100行 5列 时间步为10，能提取 100-10=90个窗口
        x = np.array(data_prepared[i:i + sequence_length, feature_dim_start_col:feature_dim_end_col+1]).flatten()  # 从第 i 行开始，提取到第 i + sequence_length 行（不包括 i + sequence_length 行）的想要的特征列， [i, i + sequence_length) 并展为一维向量 24*24 -> 1*576， 行数即对应seq_len时间步大小
        y = data_prepared[i + sequence_length, target_dim_col]  # 提取第 i + sequence_length 行（时间步）的第 target_dim_col 列作为标签。
        features.append(x)
        labels.append(y)
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def shuffle_data(features, labels):
    '''
    对特征和标签 随机打乱，帮助减少模型过拟合的风险，确保模型在训练过程中不会仅依赖于数据的顺序模式
    :param features: E.g. 90 * 50
    :param labels: E.g. 90 * 1
    :return:
    '''
    sample_num = features.shape[0]  # features.shape 返回数组的形状，features.shape[0] 返回数组的第一个维度的大小，即样本的数量
    shuffle_indices = np.random.permutation(sample_num)  # 返回一个长度为 sample_num 的随机排列的数组, 生成的数组包含从 0 到 sample_num-1 的所有整数，顺序是随机的
    features = features[shuffle_indices, :]  # 使用 随机排列的索引数组 重排（打乱）特征数组
    labels = labels[shuffle_indices]  # 使用随机排列的索引数组打乱标签数组
    return features, labels


def get_train_val_test_data(data_prepared, is_joint=False, sequence_length=20,
                            feature_dim_start_col=4, feature_dim_end_col=12, target_dim_col=2,
                            train_ratio=0.8, validate_ratio=0.1):
    '''
    单目标预测 数据集分割
    1. 选择窗口，获得feature和 label
    2. 随机打乱 feature和 label，减少模型过拟合，增加模型泛化能力
    3. 按比例生成训练集，验证集，测试集
    :param data_prepared:
    :param is_joint: 是否是联合数据集，True -> 有多个数据集 data_perpared 是三维数据；False-> 只有一个数据集，data_perpared 是二维数据
    :param sequence_length:
    :param feature_size:
    :param feature_dim_start_col:
    :param feature_dim_end_col:
    :param target_dim_col:
    :param train_ratio:
    :param validate_ratio:
    :return:
    '''
    # Feature/Label生成
    feature_size = feature_dim_end_col - feature_dim_start_col + 1
    if not is_joint:
        data_prepared = data_prepared[:, feature_dim_start_col:feature_dim_end_col + 1]  # 特征预选择
        features, labels = windows_select(data_prepared, 0, feature_size-1, sequence_length, target_dim_col) # 特征选择：默认选择所有特征，因在特征预选择时，已经选择了想要的特征，此处选特征只做二次修正
        features, labels = shuffle_data(features, labels)
    else:
        # 第一个数据集
        data_prepared[0] = data_prepared[0][:, feature_dim_start_col:feature_dim_end_col + 1]
        features, labels = windows_select(data_prepared[0], 0, feature_size-1,  sequence_length, target_dim_col)
        # 剩余数据集(从第2个数据集开始)
        for data in data_prepared[1:]:
            data = data[:, feature_dim_start_col:feature_dim_end_col + 1]  # 取对应区间维度的数据
            feature_other, label_other = windows_select(data, 0, feature_size-1,  sequence_length, target_dim_col)
            features = np.vstack((features, feature_other))  # 垂直拼接
            labels = np.hstack((labels, label_other))  # 水平拼接
        features, labels = shuffle_data(features, labels)

    print("features.shape", features.shape)
    print("labels.shape", labels.shape)

    # 数据集划分
    # 划分训练、验证和测试集
    train_row = round(features.shape[0] * train_ratio) # 训练集行数 90 * 0.8 = 72
    validate_row = round(features.shape[0] * validate_ratio) # 验证集行数 90 * 0.1 = 9
    test_row = features.shape[0] - train_row - validate_row # 测试集行数 90 - 72 - 9 = 9

    x_train = np.reshape(features[:train_row, :], (train_row, sequence_length, feature_size)) # 90 * 50 重塑为 train_row * seq_len * feature_size的数组 90 *(10 * 5)
    y_train = np.reshape(labels[:train_row], (train_row, -1)) # 90 * 1 重塑为train_row行的二维数组 90 * 1，列数自动计算

    x_val = np.reshape(features[train_row:train_row + validate_row, :], (validate_row, sequence_length, feature_size))
    y_val = np.reshape(labels[train_row:validate_row + train_row], (validate_row, -1))

    x_test = np.reshape(features[train_row + validate_row:, :], (test_row, sequence_length, feature_size))
    y_test = np.reshape(labels[train_row + validate_row:], (test_row, -1))

    print("train_samples:", x_train.shape, y_train.shape)
    print("validate_samples:", x_val.shape, y_val.shape)
    print("test_samples:", x_test.shape, y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test

'''=====================================================单目标预测数据集生成逻辑end=============================================================='''


'''=====================================================多目标预测数据集生成=============================================================='''
def windows_select_multi(data_prepared, feature_dim_start_col, feature_dim_end_col, sequence_length,
                         target_dim_start_col, target_dim_end_col):
    '''
       从输入数据中选择固定长度的时间窗口（子序列），并为每个窗口生成相应的特征和标签。每个窗口包含指定数量的时间步（sequence_length），
    并将这些时间步的特征平铺成一个一维向量。
    在时间序列数据处理中，窗口是指一段连续的时间序列数据。通过窗口化操作，可以从时间序列中提取多个固定长度的子序列，每个子序列作为模型的一个输入
    :param data_prepared: 输入的数据集，通常是一个二维数组，每行是一个时间步的特征向量
    :param feature_dim_start_col: 选择的起始特征列 默认选择所有特征
    :param feature_dim_end_col：选择的结束特征列
    :param sequence_length:时间序列的长度（窗口大小）
    :param target_dim_start_col: 预测目标维度的列开始
    :param target_dim_end_col:  预测目标维度的列结束
    :return:
    '''
    features = []  # 装有多少个feature，每个feature对应一个seq_len的特征列平铺（seq_len * 特征列数）
    labels = []
    for i in range(data_prepared.shape[0] - sequence_length):  # 可提取的窗口数量
        x = np.array(data_prepared[i:i + sequence_length, feature_dim_start_col:feature_dim_end_col+1]).flatten()  # 从第 i 行开始，提取到第 i + sequence_length 行（不包括 i + sequence_length 行）的所有列， [i, i + sequence_length) 并展为一维向量
        y = data_prepared[i + sequence_length, target_dim_start_col:target_dim_end_col+1]  # 提取第 i + sequence_length 行的第 target_dim_col 列作为标签。
        features.append(x)
        labels.append(y)
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def shuffle_data_multi(features, labels):
    '''
   对特征和标签 随机打乱，帮助减少模型过拟合，确保模型在训练过程中不会仅依赖于数据的顺序模式
   :param features: E.g. 90 * 50
   :param labels: E.g. 90 * 5
   :return:
   '''
    # shuffle data
    shuffle_indicies = np.random.permutation(features.shape[0])
    features = features[shuffle_indicies, :]
    labels = labels[shuffle_indicies, :]
    return features, labels


def get_train_val_test_data_multi(data_prepared, is_joint=False, squence_length=20,
                                  feature_dim_start_col=4, feature_dim_end_col=12, target_dim_start_col=4, target_dim_end_col=12,
                                  train_ratio=0.8, validate_ratio=0.1):
    '''
    多目标预测数据集分割
    1. 选择窗口，获得feature和 label
    2. 随机打乱 feature和 label，减少模型过拟合，增加模型泛化能力
    3. 按比例生成训练集，验证集，测试集
    :param data_prepared:
    :param is_joint: 是否是联合数据集，True -> 有多个数据集 data_perpared 是三维数据；False-> 只有一个数据集，data_perpared 是二维数据
    :param sequence_length:
    :param feature_dim_start_col:
    :param feature_dim_end_col:
    :param target_dim_end_col:
    :param target_dim_start_col
    :param train_ratio:
    :param validate_ratio:
    :return:
    '''
    # Feature/Label生成
    # data_perpared 是二维数据
    feature_size = feature_dim_end_col - feature_dim_start_col + 1
    target_size = target_dim_end_col - target_dim_start_col + 1
    if not is_joint:
        data_prepared = data_prepared[:, feature_dim_start_col:feature_dim_end_col + 1]  # 特征预选择
        features, labels = windows_select_multi(data_prepared, 0, feature_size - 1, squence_length,
                                                target_dim_start_col, target_dim_end_col)
        features, labels = shuffle_data_multi(features, labels)
    # data_perpared 是三维数据，(county_num,feature_size,temper_dim)
    else:
        data_prepared[0] = data_prepared[0][:, :feature_size]
        features, labels = windows_select_multi(data_prepared[0], 0, feature_size - 1, squence_length, target_dim_start_col, target_dim_end_col)
        for data in data_prepared[1:]:
            data = data[:, :feature_size]  # 取前面7维的数据
            # 对一个地级市内的数据进行分组
            f, l = windows_select_multi(data, 0, feature_size-1, squence_length, target_dim_start_col, target_dim_end_col)
            features = np.vstack((features, f))  # 垂直拼接
            labels = np.vstack((labels, l))  # 垂直拼接
        features, labels = shuffle_data_multi(features, labels)
    print("features.shape", features.shape)
    print("labels.shape", labels.shape)

    # 数据集划分
    # train samples
    train_row = round(features.shape[0] * train_ratio)
    validate_num = round(features.shape[0] * validate_ratio)
    test_num = features.shape[0] - train_row - validate_num
    x_train = np.reshape(features[:train_row, :], (train_row, squence_length, feature_size))
    y_train = np.reshape(labels[:train_row, :], (train_row, target_size)) # 90 * 1 重塑为train_row行的二维数组 90 * feature_size
    # validation samples
    x_val = np.reshape(features[train_row:train_row + validate_num, :], (validate_num, squence_length, feature_size))
    y_val = np.reshape(labels[train_row:train_row + validate_num, :], (validate_num, target_size))
    # test samples
    x_test = np.reshape(features[train_row + validate_num:, :], (test_num, squence_length, feature_size))
    y_test = np.reshape(labels[train_row + validate_num:, :], (test_num, target_size))
    print("train_samples:", x_train.shape, y_train.shape)
    print("validate_samples:", x_val.shape, y_val.shape)
    print("test_samples:", x_test.shape, y_test.shape)
    return (x_train, y_train, x_val, y_val, x_test, y_test)

'''=====================================================多目标预测数据集生成逻辑end=============================================================='''

def divide_data_by_geographic():
    # 按地区，划分数据集
    # 加载数据集
    file_path = '../datasets/weather/weather1_by_day/clean_weather_2013_2017_china.csv'
    data = pd.read_csv(file_path)

    # 划分三个城市的数据集
    chongqing_data = data[data['city'] == '重庆市']
    beijing_data = data[data['city'] == '北京市']
    shanghai_data = data[data['city'] == '上海市']

    # 保存划分后的数据集
    chongqing_data.to_csv('../datasets/weather/weather1_by_day/chongqing_clean_weather.csv', index=False)
    beijing_data.to_csv('../datasets/weather/weather1_by_day/beijing_clean_weather.csv', index=False)
    shanghai_data.to_csv('../datasets/weather/weather1_by_day/shanghai_clean_weather.csv', index=False)
    return beijing_data, shanghai_data, chongqing_data

def mean_sameday_samecity_data(city, data):
    '''
    对同一天同一个城市的所有数据列求平均
    :param city: 哪个城市的数据
    :param data: 数据集 Pandas的DataFrame类实例
    :return:
    '''
    # 加载数据集
    # file_path = f'../datasets/weather/weather1_by_day/{city}_clean_weather.csv'
    # data = pd.read_csv(file_path)

    # 对同一天同一城市的所有列求平均
    # groupby方法用于对DataFrame进行分组，按照date和city两个列进行分组，具有相同date和city值的行将被分在同一个组中
    # mean对每个组计算其数值列的均值（平均值）,非数值型的列，将会被忽略
    # reset_index方法用于重置索引,将date和city列从索引中移回为普通列
    grouped_data = data.groupby(['date', 'city']).mean().reset_index()

    # 填充缺失值
    # 使用插值法填充缺失值
    grouped_data.interpolate(method='linear', inplace=True)

    # 保存结果
    grouped_data.to_csv(f'../datasets/weather/weather1_by_day/divide_by_city/average_weather_by_day_{city}.csv', index=False)
    return grouped_data

def feature_expand(city, data):
    '''
    特征扩展：将同一天内不同区的数据作为额外特征，而不是简单地聚合。为每个区创建单独的特征，然后将这些特征一起输入到模型中
    :param city: 哪个城市的数据
    :param data: 数据集 Pandas的DataFrame类实例
    :return:
    '''
    # 加载数据集
    # file_path =  f'../datasets/weather/weather1_by_day/{city}_clean_weather.csv'
    # data = pd.read_csv(file_path)

    # 选择需要的列（例如日期、区、温度等）
    selected_columns = ['date', 'county', 'pressure', 'wind_speed','wind_direction','temperature', 'humidity',
                        'rain20', 'rain08','cloud','visibility',
                        'sunny', 'cloudy', 'rain', 'fog', 'haze', 'dust', 'thunder', 'lightning', 'snow', 'hail', 'wind',
                        'year', 'month', 'week', 'quarter', 'day']

    # 只保留所需列
    data = data[selected_columns]

    # 对数据进行透视，变成以日期为索引，各区的数据作为列
    # 得 date 列成为索引（即每一行代表一天），county 列中的每个唯一值（即每个区）成为新的列。对于每个日期和区的组合，
    # 聚合函数 aggfunc='mean' 会计算所有观测值的平均值。如果同一天同一个county有多个观测值，它们会被平均。
    # 比如，20130101天的重庆city的巴南市county有多行值，这多行值会被平均
    pivot_data = data.pivot_table(index='date', columns='county', aggfunc='mean')

    # 重新调整列名
    pivot_data.columns = [f"{var}_{county}" for var, county in pivot_data.columns] # temperature_闵行区, humidity_徐汇区
    pivot_data.reset_index(inplace=True) # 将 date 索引重置为普通列，使数据框的结构更像传统的二维表格

    # 从原始数据中提取日期信息
    date_info_columns = ['date', 'year', 'month', 'week', 'quarter', 'day']
    date_info = data[date_info_columns].drop_duplicates(subset=['date'])

    # 将日期信息合并到透视后的数据中
    # 指定使用 date 列作为合并的键，把从原始数据中提取的日期信息数据date_info合到透视后的数据框pivot_data中
    # 合并方式为左连接（left join），即保留 pivot_data 中所有的行，并将 date_info 中匹配的列合并进来。如果 date_info 中有 date 列与 pivot_data 中的 date 列匹配不上，对应的日期信息列会用 NaN 填充。
    pivot_data = pivot_data.merge(date_info, on='date', how='left')

    # 按照 selected_columns 中的顺序对列进行排序
    ordered_columns = (['date', 'year', 'month', 'week', 'quarter', 'day'] +
                       [f"{var}_{county}" for var in selected_columns if var != 'date' and var != 'county'
                        and var != 'year' and var != 'month' and var != 'week' and var != 'quarter' and var != 'day'
                        for county in data['county'].unique()])
    pivot_data = pivot_data[ordered_columns]

    # 填充缺失值
    # 使用插值法填充缺失值
    pivot_data.interpolate(method='linear', inplace=True)

    # 保存结果
    pivot_data.to_csv(f'../datasets/weather/weather1_by_day/divide_by_city/expanded_features_weather_by_day_{city}.csv', index=False)

    return pivot_data

def feature_engineering(geo_dataset_list):
    '''
    特征工程
    :param geo_dataset_list: 按 beijing_data, shanghai_data, chongqing_data 城市顺序的 不同城市的数据集
    :return:
    '''
    # 特征提取feature_extraction
    avg_data_list = []
    feature_expand_data_list = []
    cities = ['beijing', 'shanghai', 'chongqing']
    for data, city in zip(geo_dataset_list, cities):
        # 同天同city，特征求平均
        avg_data = mean_sameday_samecity_data(city, data) # 多元时序数据特征求平均
        # 同天同city不同county特征扩展
        feature_expand_data = feature_expand(city, data) # 特征扩展，将同一时间步（同一天）内不同地区的数据作为额外的特征
        avg_data_list.append(avg_data)
        feature_expand_data_list.append(feature_expand_data)
    return avg_data_list, feature_expand_data_list

def feature_engineering_for_electricity_and_save(electricity_dataset):
    '''
    9_LTSF_dataset 9大时间序列数据集中 电力数据集electricity的定制特征工程，主要做时间特征的提取，特征扩展feature_expand，从date列中，尽可能多的提取时序特征列
    :param electricity_dataset:
    :return:
    '''
    # 将 'date' 列转换为 datetime 类型
    electricity_dataset['date'] = pd.to_datetime(electricity_dataset['date'])

    # 提取年、月、日、小时等时间特征列
    electricity_dataset['year'] = electricity_dataset['date'].dt.year
    electricity_dataset['month'] = electricity_dataset['date'].dt.month
    electricity_dataset['day'] = electricity_dataset['date'].dt.day
    electricity_dataset['hour'] = electricity_dataset['date'].dt.hour
    # electricity_dataset['minute'] = electricity_dataset['date'].dt.minute
    electricity_dataset['day_of_week'] = electricity_dataset['date'].dt.dayofweek
    electricity_dataset['day_of_year'] = electricity_dataset['date'].dt.dayofyear
    electricity_dataset['week_of_year'] = electricity_dataset['date'].dt.isocalendar().week
    electricity_dataset['quarter'] = electricity_dataset['date'].dt.quarter

    # 将时间特征列移动到数据集最左边
    time_features = electricity_dataset[['year', 'month', 'day', 'hour', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter']]
    electricity_dataset = electricity_dataset.drop(columns=['date', 'year', 'month', 'day', 'hour', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter'])
    electricity_dataset = pd.concat([time_features, electricity_dataset], axis=1)

    # 存储
    electricity_dataset.to_csv('../datasets/9_LTSF_dataset/processed/electricity_processed.csv', index=False)

    return electricity_dataset



if __name__ == '__main__':
    pass
