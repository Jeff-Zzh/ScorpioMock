class ModelConfig:
    def __init__(self):
        self.model = 'MPDLinear_SOTA'
        self.batch_size = 32 # 在一次训练迭代中同时处理的时间序列样本数量
        self.seq_len = 30  # 输入序列长度
        self.enc_in = 9  # 输入通道数（特征列数）（根据具体的numeric_features数量调整）
        self.pred_len = 1  # 预测序列长度
        self.individual = False  # 是否为每个通道（特征列数）单独应用线性模型
        self.num_epochs = 10
        self.es_patience = 5
        self.es_verbose = True
        self.es_delta = 0.00001
        self.es_path = 'current_best_checkpoint.pt'
        self.decomposition_kernel_size = 25
        self.learning_rate = 0.001
        self.scaling_method = 'standardization' # 缩放方法: 标准化/归一化


    def __str__(self):
        return (f"ModelConfig("
                f"model={self.model}, "
                f"batch_size={self.batch_size}, "
                f"seq_len={self.seq_len}, "
                f"enc_in={self.enc_in}, "
                f"pred_len={self.pred_len}, "
                f"individual={self.individual}, "
                f"num_epochs={self.num_epochs},"
                f"es_patience={self.es_patience}, "
                f"es_verbose={self.es_verbose}, "
                f"es_delta={self.es_delta}, "
                f"es_path={self.es_path},"
                f"decomposition_kernel_size={self.decomposition_kernel_size},"
                f"learning_rate={self.learning_rate},"
                f"scaling_method={self.scaling_method})")

    # @property
    # def batch_size(self):
    #     return self.batch_size
    #
    # @batch_size.setter
    # def batch_size(self, batch_size):
    #     self.batch_size = batch_size
    #
    # @property
    # def seq_len(self):
    #     return self.seq_len
    #
    # @seq_len.setter
    # def seq_len(self, seq_len):
    #     self.seq_len = seq_len
    #
    # @property
    # def pred_len(self):
    #     return self.pred_len
    #
    # @pred_len.setter
    # def pred_len(self, pred_len):
    #     self.pred_len = pred_len
    #
    # @property
    # def individual(self):
    #     return self.individual
    #
    # @individual.setter
    # def individual(self, individual):
    #     self.individual = individual
    #
    # @property
    # def enc_in(self):
    #     return self.enc_in
    #
    # @enc_in.setter
    # def enc_in(self, enc_in):
    #     self.enc_in = enc_in


