# coding=gb2312
import numpy as np


class EvolutionStrategy:
    """
    自然进化策略 (NES) 优化算法，使用 multiprocessing.Pool 进行并行评估。
    """

    # __init__ 现在接收一个 'pool' 对象，移除了 mlp_model
    def __init__(self, initial_weights, reward_function, pool,
                 population_size=50, sigma=0.1, learning_rate=0.05, num_iterations=100):
        self.weights = initial_weights.copy()
        self.reward_function = reward_function  # 这是 main.py 中的顶级工作函数
        self.pool = pool  # 存储进程池
        self.pop_size = population_size
        self.sigma = sigma
        self.lr = learning_rate
        self.num_iterations = num_iterations

    def run(self):
        best_weights = self.weights.copy()
        best_reward = -np.inf

        for iteration in range(self.num_iterations):
            noises = np.random.randn(self.pop_size, self.weights.size)
            candidates = [self.weights + self.sigma * n for n in noises]

            # 使用进程池并行评估所有候选者
            # 将原来耗时的 for 循环替换为一行 pool.map 调用
            print(
                f"\n--- [主进程] 开始第 {iteration + 1}/{self.num_iterations} 代评估 (分发 {self.pop_size} 个任务)... ---",
                flush=True)
            rewards = np.array(self.pool.map(self.reward_function, candidates))

            print(f"--- [主进程] 第 {iteration + 1} 代评估完成。开始更新权重... ---", flush=True)

            # 更新 best
            max_idx = int(np.argmax(rewards))
            if rewards[max_idx] > best_reward and np.isfinite(rewards[max_idx]):
                best_reward = float(rewards[max_idx])
                # 直接从 candidates 数组中获取最佳权重
                best_weights = candidates[max_idx].copy()

            # 对 rewards 做稳健标准化
            finite_mask = np.isfinite(rewards)
            if np.any(finite_mask):
                finite_rewards = rewards[finite_mask]
                mean = float(np.mean(finite_rewards))
                std = float(np.std(finite_rewards))
            else:
                mean = 0.0
                std = 1.0

            if not np.isfinite(std) or std < 1e-8:
                std = 1.0

            norm_rewards = (rewards - mean) / std

            # 计算近似梯度（向量化）
            weighted_noises = np.dot(norm_rewards, noises)
            grad = weighted_noises / (self.pop_size * self.sigma)

            # 更新权重
            self.weights = self.weights + self.lr * grad

            # 打印进度
            print("迭代 {}/{}，最佳奖励 = {:.4f}，当前平均奖励 = {:.4f}".format(
                iteration + 1, self.num_iterations, float(best_reward if np.isfinite(best_reward) else -np.inf),
                mean
            ))

        return best_weights, best_reward
