class CommunicationEnergyModel:
    def __init__(self, E_o=0.5, E_ele=50e-9, E_fs=10e-12, E_mp=0.0013e-12, E_DA=5e-9, d_t=87):
        self.E_o = E_o  # 传感器节点初始能量
        self.E_ele = E_ele # 每bit发射和接收的能量
        self.E_fs = E_fs # 发射功率放大器的能量消耗 - 自由空间模型
        self.E_mp = E_mp # 发射功率放大器的能量消耗 - 多路径衰减模型
        self.E_DA = E_DA # 每bit数据聚合能耗
        self.d_t = d_t # 距离阈值 - 传感节点与接收节点之间的据里d小于阈值d_t时，使用自由空间模型；否则，使用多路径衰减模型

    def transmission_energy(self, l, d):
        '''
        发送-能量消耗
        :param l: 消息bit长度
        :param d: 传输两节点间的距离
        :return:
        '''
        if d < self.d_t:
            E_amp = l * self.E_fs * (d ** 2)
        else:
            E_amp = l * self.E_mp * (d ** 4)
        return (l * self.E_ele) + E_amp # 发送 + 发送功率放大器 能耗

    def reception_energy(self, l):
        '''
        接收-能量消耗
        :param l:
        :return:
        '''
        return l * self.E_ele

    def processing_energy(self, l, N=1, is_edge_node=False):
        '''
        处理数据-能量消耗
        :param l: 要聚合的bit长度
        :param N: 簇头的簇成员数- 边节点连接的端节点数量
        :param is_edge_node:
        :return:
        '''
        if is_edge_node:
            return l * N * self.E_DA
        return l * self.E_DA

    def total_energy(self, l, d, N=1, is_edge_node=False):
        '''
        总能量消耗 -
        :param l:  bit长度
        :param d:  终端设备与边节点之间的距离（米）
        :param N:  边节点连接的端节点数量
        :param is_edge_node:
        :return:
        '''
        if is_edge_node:
            E_Rx = self.reception_energy(l * N) # 接收端侧数据能耗
            E_Px = self.processing_energy(l, N, is_edge_node) # 处理数据能耗
            E_Tx = self.transmission_energy(l, d) # 发送给（云）数据能耗
            return E_Rx + E_Tx + E_Px
        else: # 端节点
            E_Tx = self.transmission_energy(l, d) # 发送给（云）数据能耗
            E_Px = self.processing_energy(l) # 传输能量
            return E_Tx + E_Px


# 示例用法
communication_energy_model = CommunicationEnergyModel()

# 示例值：一个终端设备（不是边节点）
l = 4000  # 消息长度（比特）
d = 50  # 终端设备与边节点之间的距离（米）

total_energy_sensor = communication_energy_model.total_energy(l, d)
print(f"终端设备总能量消耗: {total_energy_sensor} J")

# 示例值：一个边节点
N = 10  # 边节点连接的终端设备数量
total_energy_edge_node = communication_energy_model.total_energy(l, d, N, is_edge_node=True)
print(f"边节点总能量消耗: {total_energy_edge_node} J")
