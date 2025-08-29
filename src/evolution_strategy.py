# coding=gb2312
import numpy as np
import torch

class EvolutionStrategy:
    """
    ��Ȼ�������� (NES) �Ż��㷨 ? ���Ӷ� NaN/inf �ķ�����������ʵ�������³���Ժ�Ч�ʡ�
    """
    def __init__(self, initial_weights, reward_function, mlp_model,
                 population_size=50, sigma=0.1, learning_rate=0.01, num_iterations=100):
        # initial_weights ������ numpy ���飬�� mlp_model �ڲ��� PyTorch ����
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
            # ���� noises ����shape = (pop_size, n_weights)
            # noises ��Ȼ�� numpy ���飬��Ϊ������ weights �����ǻ��� numpy ��
            noises = np.random.randn(self.pop_size, self.weights.size)
            candidates = self.weights + self.sigma * noises

            rewards = np.full(self.pop_size, -1e9, dtype=float)  # Ԥ��伫��ֵ��Ϊ����

            # ����ÿ����ѡ
            for i in range(self.pop_size):
                try:
                    # reward_function ���� numpy ������ΪȨ������
                    r = float(self.reward_function(candidates[i], self.mlp_model))
                    if not np.isfinite(r):
                        r = -1e9
                except Exception as e:
                    r = -1e9
                rewards[i] = r

            # ���� best
            max_idx = int(np.argmax(rewards))
            if rewards[max_idx] > best_reward and np.isfinite(rewards[max_idx]):
                best_reward = float(rewards[max_idx])
                best_weights = (self.weights + self.sigma * noises[max_idx]).copy()

            # �� rewards ���Ƚ���׼������ɸѡ����ֵ
            finite_mask = np.isfinite(rewards)
            if np.any(finite_mask):
                finite_rewards = rewards[finite_mask]
                mean = float(np.mean(finite_rewards))
                std = float(np.std(finite_rewards))
            else:
                mean = 0.0
                std = 1.0

            # ���� std
            if not np.isfinite(std) or std == 0.0:
                std = 1.0

            # ��׼����������Ԫ�أ�����֮ǰ��Ϊ -1e9 �ģ�
            norm_rewards = (rewards - mean) / (std + 1e-8)

            # ��������ݶȣ���������
            weighted_noises = (norm_rewards[:, None] * noises).sum(axis=0)
            grad = weighted_noises / (self.pop_size * self.sigma)

            # ����Ȩ��
            self.weights = self.weights + self.lr * grad

            # ��ӡ����
            print("���� {}/{}����ѽ��� = {:.4f}����ǰƽ������ = {:.4f}".format(
                iteration + 1, self.num_iterations, float(best_reward if np.isfinite(best_reward) else -np.inf),
                float(np.mean(rewards[np.isfinite(rewards)]) if np.any(np.isfinite(rewards)) else -np.inf)
            ))

        # ���ҵ�������Ȩ�����õ�ģ��
        try:
            if hasattr(self.mlp_model, 'set_weights'):
                self.mlp_model.set_weights(best_weights)
        except Exception as e:
            print(f"��������ģ��Ȩ��ʧ��: {e}")

        return best_weights, best_reward
