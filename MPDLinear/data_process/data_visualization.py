import os

import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

# matplotlib.rcParams['font.sans-serif']=['Droid Sans Fallback'] #用来正常显示中文标签 没有这种中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取当前脚本的目录
cur_dir = os.path.dirname(os.path.abspath(__file__))
# 图片保存目录
pic_dir = os.path.join(cur_dir, 'pic')
# 如果目录不存在，则创建目录
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

## 绘制某个属性关于季度和地理位置的变化情况
def plot_of_quarter_city(X, attribe):
    # 将属性按照季度和城市分组
    df_group = X[attribe].groupby([X['quarter'], X['city']]).mean()
    # 对多级索引中的city进行重命名
    df_group = df_group.rename({'重庆市': 'chongqing', '北京市': 'beijing', '上海市': 'shanghai'}, axis='index')
    # 将series 按照city展开成一个表格
    pop_df = df_group.unstack()
    return pop_df


def visibility(weather):
    # 每个城市在每个季度的可见度
    df_vb_group1 = plot_of_quarter_city(weather, 'visibility')
    # 对多级索引重新命名
    df_vb_group1.plot(kind='bar') # 生成条形图
    plt.title("可见度变化情况")
    # 保存图像到本地
    plt.savefig(os.path.join(pic_dir, 'visibility.png'))
    plt.show()


def rain20_rain08(weather):
    # 绘制降雨量
    df_rain20_group = plot_of_quarter_city(weather, 'rain20')
    df_rain20_group.plot(kind='bar')
    plt.title("20时降雨量")
    plt.savefig(os.path.join(pic_dir, 'rain20.png'))
    df_rain08_group = plot_of_quarter_city(weather, 'rain08')
    df_rain08_group.plot(kind='bar')
    plt.title("08时降雨量")
    plt.savefig(os.path.join(pic_dir, 'rain08.png'))
    plt.show()


def tempature(weather):
    ## 温度变化情况
    df_tempature_group = plot_of_quarter_city(weather, 'temperature')
    df_tempature_group.plot(kind='bar')
    plt.title("温度变化情况")
    plt.savefig(os.path.join(pic_dir, 'tempature.png'))
    plt.show()


def humidity(weather):
    ## 湿度变化情况
    df_humidity_group = plot_of_quarter_city(weather, 'humidity')
    df_humidity_group.plot(kind='bar')
    plt.title("湿度变化情况")
    plt.savefig(os.path.join(pic_dir, 'humidity.png'))
    plt.show()


def pressure(weather):
    ## 气压变化情况
    df_pressure_group = plot_of_quarter_city(weather, 'pressure')
    df_pressure_group.plot(kind='bar')
    plt.title("气压变化情况")
    plt.savefig(os.path.join(pic_dir, 'pressure.png'))
    plt.show()


def wind_speed(weather):
    ## 风速变化情况
    df_wind_speed_group = plot_of_quarter_city(weather, 'wind_speed')
    df_wind_speed_group.plot(kind='bar')
    plt.title("风速变化情况")
    plt.savefig(os.path.join(pic_dir, 'wind_speed.png'))
    plt.show()

def wind_direction(weather):
    ## 最大风速的风向(度)变化情况
    df_wind_direction_group = plot_of_quarter_city(weather, 'wind_direction')
    df_wind_direction_group.plot(kind='bar')
    plt.title("最大风速的风向(度)")
    plt.savefig(os.path.join(pic_dir, 'wind_direction.png'))
    plt.show()


def cloud(weather):
    ## 云量变化情况
    df_cloud_group = plot_of_quarter_city(weather, 'cloud')
    df_cloud_group.plot(kind='bar')
    plt.title("云量变化情况")
    plt.savefig(os.path.join(pic_dir, 'cloud.png'))
    plt.show()


attributes = ['pressure', 'wind_speed', 'wind_direction', 'temperature', 'humidity', 'rain20', 'rain08', 'cloud', 'visibility']
def correlation_scatter_matrix(weather, attributes):
    # 相关性散点图 展示各个属性相互之间的相关性
    # scatter_matrix(weather[attributes], figsize=(12, 8))
    # 绘制散点矩阵图，并将对角线直方图设为单独的颜色
    sns.pairplot(weather[attributes], diag_kind='kde', plot_kws={'alpha': 0.6}, diag_kws={'color': 'red'})
    plt.savefig(os.path.join(pic_dir, '相关性散点图correlation_scatter_matrix.png'))
    # plt.show()


def correlation_pic(weather, attributes):
    # 计算属性之间的相关系数矩阵
    correlation_matrix = weather[attributes].corr()
    # print("\n相关系数矩阵：")
    # print(correlation_matrix)
    # 相关系数热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(pic_dir, '相关系数热力图correlation_matrix.png'))
    plt.show()

def draw_all(weather, attributes):
    # 按照 季度 和 城市 分组的 平均值直方图
    pressure(weather)
    wind_speed(weather)
    wind_direction(weather)
    tempature(weather)
    humidity(weather)
    rain20_rain08(weather)
    cloud(weather)
    visibility(weather)

    # 相关性
    # 散点矩阵图
    correlation_scatter_matrix(weather, attributes)
    # 相关系数热力图
    correlation_pic(weather, attributes)
