from CommunicationEnergyModel import CommunicationEnergyModel
from DataProcessingEnergyModel import DataProcessingEnergyModel
import config

# 示例使用
comm_energy_model = CommunicationEnergyModel()
proc_energy_model = DataProcessingEnergyModel()

l = config.TEN_MB_BITS  # 消息长度 bit
model_hyper_param_l = config.ONE_MB_BITS  # 模型超参数bit位数
end_edge_distance = 5  # 终端设备与边缘节点之间的距离（米）
edge_cloud_distance = 5000000  # 边缘节点与云服务器之间的距离（米）
N = 10  # 边缘节点连接的终端设备数量
M = 3  # 云节点连接的边缘节点数量
T_idle = 10  # 空闲状态持续时间（秒）
T_sleep = 20  # 睡眠状态持续时间（秒）
T_wake = 1  # 唤醒状态持续时间（秒）

l = config.ONE_MB_BITS  # 模型inference需要除了爹输入数据的总bit
l_pred = config.ONE_MB_BITS // 10  # 预测数据量（比特）
l_real = config.ONE_MB_BITS # 真实数据量（比特）
l_model = config.MB_100_BITS  # 模型训练数据量（比特）

# 终端设备总能量消耗
total_comm_energy_sensor = comm_energy_model.total_energy(l, model_hyper_param_l, end_edge_distance,
                                                          edge_cloud_distance, N, M, T_idle=T_idle, T_sleep=T_sleep,
                                                          T_wake=T_wake, is_edge_node=False, is_cloud_node=False)
total_proc_energy_sensor = proc_energy_model.sensor_node_energy(l_real)
total_energy_sensor = total_comm_energy_sensor + total_proc_energy_sensor
print(f"终端设备总能量消耗: {total_energy_sensor} J")

# 边缘节点总能量消耗
total_comm_energy_edge = comm_energy_model.total_energy(l, model_hyper_param_l, end_edge_distance,
                                                        edge_cloud_distance, N, M, T_idle=T_idle, T_sleep=T_sleep,
                                                        T_wake=T_wake, is_edge_node=True, is_cloud_node=False)
total_proc_energy_edge = proc_energy_model.edge_node_energy(l, l_pred, l_real)
total_energy_edge_node = total_comm_energy_edge + total_proc_energy_edge
print(f"边缘节点总能量消耗: {total_energy_edge_node} J")

# 云端节点总能量消耗
total_comm_energy_cloud = comm_energy_model.total_energy(l, model_hyper_param_l, end_edge_distance,
                                                         edge_cloud_distance, N, M, T_idle=T_idle, T_sleep=T_sleep,
                                                         T_wake=T_wake, is_edge_node=False, is_cloud_node=True)

total_proc_energy_cloud = proc_energy_model.cloud_node_energy(l, l_pred, l_real, l_model)
total_energy_cloud_node = total_comm_energy_cloud
print(f"云端节点总能量消耗: {total_energy_cloud_node} J")
