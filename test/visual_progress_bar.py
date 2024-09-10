import time
from tqdm import tqdm

# 示例循环，使用 tqdm 显示进度条
for i in tqdm(range(100), desc="Processing"):
    # 模拟一些处理
    time.sleep(0.1)  # 每次循环等待 0.1 秒