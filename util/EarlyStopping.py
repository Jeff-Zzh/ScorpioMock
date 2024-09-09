import torch


class EarlyStopping:
    def __init__(self, logger, patience=5, verbose=False, delta=0, path='current_best_checkpoint.pt'):
        """
        通过监控验证集的损失 (val_loss)，EarlyStopping 可以在指定的 patience（容忍度）内检测到验证损失不再改善时停止训练。
        delta 参数控制验证损失的最小变化量。
        path 参数指定保存最佳模型权重的路径。
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_score = None
        self.logger = logger # 日志记录

    def __call__(self, val_loss, model):
        '''
        使EarlyStopping像函数一样被调用
        :param val_loss: 训练一个 epoch 之后计算的验证集损失
        :param model: 是当前训练的模型实例，继承自 torch.nn.Module
        :return:
        '''
        score = -val_loss # score 是验证损失的负值，因为我们希望损失越低越好，所以通过取负号来将其转化为一个越大越好的分数。 val_loss是>0的

        if self.best_score is None: # 第一次调用EarlyStopping
            self.best_score = score
            self.save_checkpoint(val_loss, model) # 将当前模型保存为最佳模型的状态
        elif score < self.best_score + self.delta: # 如果当前的 score 比 best_score 小，且与 best_score 之差没有超过 delta（即损失没有明显改善），就会增加 counter（计数器）来记录未改善的 epoch 次数
            self.counter += 1
            if self.verbose:
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: # 如果 counter 达到了 patience（容忍的最大次数），则将 early_stop 标志设置为 True，指示应该停止训练
                self.early_stop = True
        else: # 如果当前的 score 比之前的 best_score 更大（即验证损失减少了），则更新 best_score 为当前的 score
            self.best_score = score
            self.save_checkpoint(val_loss, model) # 并且保存当前模型状态，因为这是新的最佳状态
            self.counter = 0 # 同时将 counter 重新置为 0，因为在这个 epoch 中验证集损失有了改善

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            if self.best_loss is None:
                self.logger.info(f'Validation loss decreased for the first time. Saving model ...')
                print(f'Validation loss decreased for the first time. Saving model ...')
            else:
                self.logger.info( f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss
