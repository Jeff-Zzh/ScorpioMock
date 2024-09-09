
def run():
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