# coding=gb2312
import numpy as np
import torch
from tqdm import tqdm  # 步骤1：在这里导入tqdm


class EvolutionStrategy:
    """
    自然进化策略 (NES) 优化算法 ? 增加对 NaN/inf 的防护与向量化实现以提高鲁棒性和效率。
    """

    def __init__(self, initial_weights, reward_function, mlp_model,
                 population_size=50, sigma=0.1, learning_rate=0.01, num_iterations=100):
        self.weights = initial_weights.copy()
        self.reward_function = reward_function
        self.mlp_model = mlp_model
        self.pop_size = population_size
        self.sigma = sigma
        self.lr = learning_rate
        self.num_iterations = num_iterations

    def run(self):
        best_weights = self.weights.copy()
        best_reward = -np.inf

        # 主循环（迭代/代数）
        for iteration in range(self.num_iterations):
            # 生成 noises 矩阵
            noises = np.random.randn(self.pop_size, self.weights.size)
            candidates = self.weights + self.sigma * noises

            # ==================== tqdm 修改部分 ====================
            # 步骤2：在评估种群的循环中加入tqdm
            rewards_list = []
            # tqdm包裹了迭代器candidates，desc是进度条的描述
            # leave=False 表示此进度条完成后会从终端消失，保持界面整洁
            for candidate in tqdm(candidates, desc=f"迭代 {iteration + 1}/{self.num_iterations}", leave=False):
                try:
                    r = float(self.reward_function(candidate, self.mlp_model))
                    if not np.isfinite(r):
                        r = -1e9  # 如果奖励值是inf或nan，则设为一个极低的兜底值
                except Exception as e:
                    r = -1e9  # 如果计算过程中出现任何异常，也设为兜底值
                rewards_list.append(r)

            rewards = np.array(rewards_list)  # 将评估结果转换为numpy数组
            # ======================================================

            # 更新 best
            max_idx = int(np.argmax(rewards))
            if rewards[max_idx] > best_reward and np.isfinite(rewards[max_idx]):
                best_reward = float(rewards[max_idx])
                best_weights = (self.weights + self.sigma * noises[max_idx]).copy()

            # 对 rewards 做稳健标准化
            finite_mask = np.isfinite(rewards)
            if np.any(finite_mask):
                finite_rewards = rewards[finite_mask]
                mean = float(np.mean(finite_rewards))
                std = float(np.std(finite_rewards))
            else:
                mean = 0.0
                std = 1.0

            if not np.isfinite(std) or std == 0.0:
                std = 1.0

            norm_rewards = (rewards - mean) / (std + 1e-8)

            # 计算近似梯度
            weighted_noises = (norm_rewards[:, None] * noises).sum(axis=0)
            grad = weighted_noises / (self.pop_size * self.sigma)

            # 更新权重
            self.weights = self.weights + self.lr * grad

            # 打印每代（iteration）的最终总结
            # 因为tqdm的进度条已经显示了内部循环的进度，所以这里只保留总结信息
            print("迭代 {}/{} | 最高奖励: {:.4f} | 平均奖励: {:.4f}".format(
                iteration + 1, self.num_iterations,
                float(best_reward if np.isfinite(best_reward) else -np.inf),
                float(np.mean(rewards[np.isfinite(rewards)]) if np.any(np.isfinite(rewards)) else -np.inf)
            ))

        # 将找到的最优权重设置到模型
        try:
            if hasattr(self.mlp_model, 'set_weights'):
                self.mlp_model.set_weights(best_weights)
        except Exception as e:
            print(f"设置最终模型权重失败: {e}")

        return best_weights, best_reward