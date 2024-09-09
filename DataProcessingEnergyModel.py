class DataProcessingEnergyModel:
    def __init__(self, E_train=2e-8, E_infer=1e-8, E_store=5e-9, E_metric=1e-9, E_flow=2e-9, E_collect=3e-9):
        # 初始化数据处理能量模型参数
        self.E_train = E_train  # 每bit数据训练模型的能量消耗（Joule/bit, J/bit）
        self.E_infer = E_infer  # 每bit数据推理的能量消耗（Joule/bit, J/bit）
        self.E_store = E_store  # 每bit数据存储的能量消耗（Joule/bit, J/bit）
        self.E_metric = E_metric  # 动态误差度量策略的能量消耗（Joule/bit, J/bit）
        self.E_flow = E_flow  # 数据流转策略的能量消耗（Joule/bit, J/bit）
        self.E_collect = E_collect  # 传感器收集数据的能量消耗（Joule/bit, J/bit）

    def cloud_node_energy(self, l, l_pred, l_real, l_model):
        # 计算云节点能量消耗
        E_train = l_model * self.E_train  # 训练新的数据预测模型能耗
        E_infer = l * self.E_infer  # 模型正常serving，推理能耗
        E_store = (l_pred + l_real) * self.E_store  # 数据存储能耗（预测数据+真实数据）
        return E_train + E_infer + E_store

    def edge_node_energy(self, l, l_pred, l_real):
        # 计算边缘节点能量消耗
        E_infer = l * self.E_infer  # 模型正常serving，推理能耗
        E_metric = l * self.E_metric  # 动态误差度量策略运行能耗
        E_flow = l * self.E_flow  # 数据流转策略运行能耗
        E_store = (l_pred + l_real) * self.E_store  # 数据存储能耗（预测数据+真实数据）
        return E_infer + E_metric + E_flow + E_store

    def sensor_node_energy(self, l_real):
        # 计算终端设备能量消耗
        E_collect = l_real * self.E_collect  # 传感器收集数据能耗
        E_store = l_real * self.E_store  # 数据存储能耗（真实数据）
        return E_collect + E_store
