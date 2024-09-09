from CommunicationEnergyModel import CommunicationEnergyModel
from DataProcessingEnergyModel import DataProcessingEnergyModel

comm_energy_model = CommunicationEnergyModel()
proc_energy_model = DataProcessingEnergyModel()

l = 16 * 1024 * 1024  # 16 MB 消息长度 bit
model_hyper_param_l = 2 * 1024 * 1024  # 2 MB 模型超参数 bit
end_edge_distance = 5  # 终端设备与边缘节点之间的距离（米）
edge_cloud_distance = 500000  # 边缘节点与云服务器之间的距离（米）
N = 20  # 边缘节点连接的终端设备数量
M = 6  # 云节点连接的边缘节点数量
T_idle = 20  # 空闲状态持续时间（秒）
T_sleep = 40  # 睡眠状态持续时间（秒）
T_wake = 3  # 唤醒状态持续时间（秒）

l_infer = 4 * 1024 * 1024  # 4 MB 模型inference需要处理的数据 bit
l_pred = 300 * 1024 * 8  # 300 KB 预测数据量 bit
l_real = 2 * 1024 * 1024  # 2 MB 真实数据量 bit
l_model = 100 * 1024 * 1024  # 100 MB 模型训练数据量 bit


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
total_energy_cloud_node = total_comm_energy_cloud + total_proc_energy_cloud
print(f"云端节点总能量消耗: {total_energy_cloud_node} J")