import logging
import os
from datetime import datetime

# 创建日志记录器
def setup_logger(log_dir, model_name, config):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 生成唯一的日志文件名
    log_filename = (f"{model_name}-{current_time}-bs{config.batch_size}-sl{config.seq_len}-pl{config.pred_len}-enc{config.enc_in}" +
                    f"-individual{config.individual}-num_epochs{config.num_epochs}-es_patience{config.es_patience}" +
                    f"-es_verbose{config.es_verbose}-es_delta{config.es_delta}-es_path{config.es_path}-ks{config.decomposition_kernel_size}" +
                    f"-lr{config.learning_rate}-{config.scaling_method}-{config.device}.log")
    log_filepath = os.path.join(log_dir, log_filename)

    # 配置日志记录器
    logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    return logger, log_filename, log_filepath

