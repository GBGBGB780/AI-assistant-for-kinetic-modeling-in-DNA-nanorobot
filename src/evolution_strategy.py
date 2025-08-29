# coding=gb2312
import numpy as np
import torch

class EvolutionStrategy:
    """
    自然进化策略 (NES) 优化算法 ? 增加对 NaN/inf 的防护与向量化实现以提高鲁棒性和效率。
    """
    def __init__(self, initial_weights, reward_function, mlp_model,
                 population_size=50, sigma=0.1, learning_rate=0.01, num_iterations=100):
        # initial_weights 现在是 numpy 数组，但 mlp_model 内部是 PyTorch 张量
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

        for iteration in range(self.num_iterations):
            # 生成 noises 矩阵：shape = (pop_size, n_weights)
            # noises 仍然是 numpy 数组，因为后续的 weights 更新是基于 numpy 的
            noises = np.random.randn(self.pop_size, self.weights.size)
            candidates = self.weights + self.sigma * noises

            rewards = np.full(self.pop_size, -1e9, dtype=float)  # 预填充极低值作为兜底

            # 评估每个候选
            for i in range(self.pop_size):
                try:
                    # reward_function 期望 numpy 数组作为权重输入
                    r = float(self.reward_function(candidates[i], self.mlp_model))
                    if not np.isfinite(r):
                        r = -1e9
                except Exception as e:
                    r = -1e9
                rewards[i] = r

            # 更新 best
            max_idx = int(np.argmax(rewards))
            if rewards[max_idx] > best_reward and np.isfinite(rewards[max_idx]):
                best_reward = float(rewards[max_idx])
                best_weights = (self.weights + self.sigma * noises[max_idx]).copy()

            # 对 rewards 做稳健标准化：先筛选有限值
            finite_mask = np.isfinite(rewards)
            if np.any(finite_mask):
                finite_rewards = rewards[finite_mask]
                mean = float(np.mean(finite_rewards))
                std = float(np.std(finite_rewards))
            else:
                mean = 0.0
                std = 1.0

            # 兜底 std
            if not np.isfinite(std) or std == 0.0:
                std = 1.0

            # 标准化（对所有元素，包括之前设为 -1e9 的）
            norm_rewards = (rewards - mean) / (std + 1e-8)

            # 计算近似梯度（向量化）
            weighted_noises = (norm_rewards[:, None] * noises).sum(axis=0)
            grad = weighted_noises / (self.pop_size * self.sigma)

            # 更新权重
            self.weights = self.weights + self.lr * grad

            # 打印进度
            print("迭代 {}/{}，最佳奖励 = {:.4f}，当前平均奖励 = {:.4f}".format(
                iteration + 1, self.num_iterations, float(best_reward if np.isfinite(best_reward) else -np.inf),
                float(np.mean(rewards[np.isfinite(rewards)]) if np.any(np.isfinite(rewards)) else -np.inf)
            ))

        # 将找到的最优权重设置到模型
        try:
            if hasattr(self.mlp_model, 'set_weights'):
                self.mlp_model.set_weights(best_weights)
        except Exception as e:
            print(f"设置最终模型权重失败: {e}")

        return best_weights, best_reward
