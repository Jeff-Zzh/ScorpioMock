class CommunicationEnergyModel:
    def __init__(self, E_o=0.5, E_ele=50e-9, E_fs=10e-12, E_mp=0.0013e-12, E_DA=5e-9, d_t=87, E_idle=1e-6, E_sleep=1e-9,
                 E_wake=1e-7):
        # 初始化通信能量模型参数
        self.E_o = E_o  # 传感器节点的初始能量（Joule, J）
        self.E_ele = E_ele  # 每bit传输和接收的能量（Joule/bit, J/bit）
        self.E_fs = E_fs  # 自由空间模型下的发射功率放大器的能量消耗（Joule/bit/m², J/bit/m²）
        self.E_mp = E_mp  # 多路径衰减模型下的发射功率放大器的能量消耗（Joule/bit/m^4, J/bit/m^4）
        self.E_DA = E_DA  # 每bit数据聚合的能耗（Joule/bit, J/bit）
        self.d_t = d_t  # 距离阈值（m），决定使用 自由空间模型 还是 多路径衰减模型
        self.E_idle = E_idle  # 空闲状态的能量消耗（Joule, J）
        self.E_sleep = E_sleep  # 睡眠状态的能量消耗（Joule, J）
        self.E_wake = E_wake  # 唤醒状态的能量消耗（Joule, J）

    def transmission_energy(self, l, d):
        '''
        计算数据发送能量消耗
        :param l:
        :param d:
        :return:
        '''
        if d < self.d_t:
            E_amp = l * self.E_fs * (d ** 2)
        else:
            E_amp = l * self.E_mp * (d ** 4)
        return (l * self.E_ele) + E_amp

    def reception_energy(self, l):
        '''
        计算接收数据能量消耗
        :param l:
        :return:
        '''
        return l * self.E_ele

    def aggregation_energy(self, l, N=1, process_multi_node=False):
        '''
        计算数据聚合能量消耗
        :param l:
        :param N:
        :param process_multi_node:
        :return:
        '''
        if process_multi_node:
            return l * N * self.E_DA
        return l * self.E_DA

    def idle_energy(self, T_idle):
        # 计算空闲状态能量消耗
        return self.E_idle * T_idle

    def sleep_energy(self, T_sleep):
        # 计算睡眠状态能量消耗
        return self.E_sleep * T_sleep

    def wake_energy(self, T_wake):
        # 计算唤醒状态能量消耗
        return self.E_wake * T_wake

    def total_energy(self, l, model_hyper_param_l, end_edge_distance, edge_cloud_distance, N=1, M=1, T_idle=1,
                     T_sleep=1, T_wake=1, is_edge_node=False, is_cloud_node=False):
        '''
        计算总通信能量消耗
        :param l: 传输数据bit位数
        :param model_hyper_param_l: 模型超参数bit位数
        :param end_edge_distance: 终端设备与边缘节点之间的距离（米）
        :param edge_cloud_distance: 边缘节点与云服务器之间的距离（米）
        :param N:边缘节点连接的终端设备数量
        :param M:云节点连接边缘节点数量
        :param T_idle: 空闲状态持续时间（秒）
        :param T_sleep: 睡眠状态持续时间（秒）
        :param T_wake: 唤醒状态持续时间（秒）
        :param is_edge_node: 该节点是否为边缘节点
        :param is_cloud_node: 该节点是否为云节点
        :return:
        '''
        idle_energy = self.idle_energy(T_idle)
        sleep_energy = self.sleep_energy(T_sleep)
        wake_energy = self.wake_energy(T_wake)

        if is_cloud_node:
            pure_comm_energy = self.cloud_communication_energy_pure(l, M, is_cloud_node, model_hyper_param_l,
                                                                    edge_cloud_distance)
            return pure_comm_energy + idle_energy + sleep_energy + wake_energy
        if is_edge_node:
            pure_comm_energy = self.edge_communication_energy_pure(l, N, is_edge_node, end_edge_distance)
            return pure_comm_energy + idle_energy + sleep_energy + wake_energy
        else:
            pure_comm_energy = self.end_communication_energy_pure(l, end_edge_distance)
            return pure_comm_energy + idle_energy + sleep_energy + wake_energy

    def cloud_communication_energy_pure(self, l, M, is_cloud_node, model_hyper_param_l, edge_cloud_distance):
        E_Rx = self.reception_energy(l * M)  # 云节点接收 M 个边节点的数据，每个边节点的数据长度为 l
        E_Px = self.aggregation_energy(l, M, process_multi_node=is_cloud_node)  # 云节点的数据聚合能量消耗
        E_Tx = self.transmission_energy(model_hyper_param_l * M, edge_cloud_distance)  # 云节点 更新模型参数下发给边节点
        return E_Rx + E_Tx + E_Px

    def edge_communication_energy_pure(self, l, N, is_edge_node, end_edge_distance):
        E_Rx = self.reception_energy(l * N)  # 接收数据能量 边节点（簇头）要接收 N 个端节点的数据，每个端节点的数据长度为l
        E_Px = self.aggregation_energy(l, N, process_multi_node=is_edge_node)  # 数据聚合能量消耗
        E_Tx = self.transmission_energy(l, end_edge_distance)  # 数据发送能量消耗
        return E_Rx + E_Tx + E_Px

    def end_communication_energy_pure(self, l, end_edge_distance):
        E_Tx = self.transmission_energy(l, end_edge_distance)
        E_Px = self.aggregation_energy(l)
        return E_Tx + E_Px
