from collections import deque
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

N = 10  # 1个Edge Node N个 End Node

data_from_end_list = []
for i in range(N):
    data_from_end_list.append(deque())


# 多线程(N个线程)代替 端节点 1个线程代替边节点 - 线程池

# 模拟端节点的数据生成函数
def end_node_task(node_id):
    while True:
        data = random.randint(1, 100)  # 生成随机数据
        data_from_end_list[node_id].append(data)
        print(f"End Node {node_id} generated data: {data}")
        time.sleep(random.uniform(0.1, 0.5))  # 模拟数据生成的间隔


# 模拟边节点的数据处理函数
def edge_node_task():
    while True:
        for i in range(N):
            if data_from_end_list[i]:
                data = data_from_end_list[i].popleft()
                print(f"Edge Node processed data from End Node {i}: {data}")
        time.sleep(0.2)  # 模拟数据处理的间隔


# 创建线程池
with ThreadPoolExecutor(max_workers=N + 1) as executor:
    # 提交端节点任务
    end_futures = [executor.submit(end_node_task, i) for i in range(N)] # 将一个可调用对象（例如函数）及其参数提交到线程池中进行异步执行
    # 提交边节点任务
    edge_future = executor.submit(edge_node_task)

    try:
        for future in as_completed(end_futures):
            future.result()  # 获取线程执行结果（此处不会被执行，因为任务是无限循环的）
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
