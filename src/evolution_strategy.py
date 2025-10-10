# coding=gb2312
import numpy as np
import torch
from tqdm import tqdm  # ����1�������ﵼ��tqdm


class EvolutionStrategy:
    """
    ��Ȼ�������� (NES) �Ż��㷨 ? ���Ӷ� NaN/inf �ķ�����������ʵ�������³���Ժ�Ч�ʡ�
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

        # ��ѭ��������/������
        for iteration in range(self.num_iterations):
            # ���� noises ����
            noises = np.random.randn(self.pop_size, self.weights.size)
            candidates = self.weights + self.sigma * noises

            # ==================== tqdm �޸Ĳ��� ====================
            # ����2����������Ⱥ��ѭ���м���tqdm
            rewards_list = []
            # tqdm�����˵�����candidates��desc�ǽ�����������
            # leave=False ��ʾ�˽�������ɺ����ն���ʧ�����ֽ�������
            for candidate in tqdm(candidates, desc=f"���� {iteration + 1}/{self.num_iterations}", leave=False):
                try:
                    r = float(self.reward_function(candidate, self.mlp_model))
                    if not np.isfinite(r):
                        r = -1e9  # �������ֵ��inf��nan������Ϊһ�����͵Ķ���ֵ
                except Exception as e:
                    r = -1e9  # �����������г����κ��쳣��Ҳ��Ϊ����ֵ
                rewards_list.append(r)

            rewards = np.array(rewards_list)  # ���������ת��Ϊnumpy����
            # ======================================================

            # ���� best
            max_idx = int(np.argmax(rewards))
            if rewards[max_idx] > best_reward and np.isfinite(rewards[max_idx]):
                best_reward = float(rewards[max_idx])
                best_weights = (self.weights + self.sigma * noises[max_idx]).copy()

            # �� rewards ���Ƚ���׼��
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

            # ��������ݶ�
            weighted_noises = (norm_rewards[:, None] * noises).sum(axis=0)
            grad = weighted_noises / (self.pop_size * self.sigma)

            # ����Ȩ��
            self.weights = self.weights + self.lr * grad

            # ��ӡÿ����iteration���������ܽ�
            # ��Ϊtqdm�Ľ������Ѿ���ʾ���ڲ�ѭ���Ľ��ȣ���������ֻ�����ܽ���Ϣ
            print("���� {}/{} | ��߽���: {:.4f} | ƽ������: {:.4f}".format(
                iteration + 1, self.num_iterations,
                float(best_reward if np.isfinite(best_reward) else -np.inf),
                float(np.mean(rewards[np.isfinite(rewards)]) if np.any(np.isfinite(rewards)) else -np.inf)
            ))

        # ���ҵ�������Ȩ�����õ�ģ��
        try:
            if hasattr(self.mlp_model, 'set_weights'):
                self.mlp_model.set_weights(best_weights)
        except Exception as e:
            print(f"��������ģ��Ȩ��ʧ��: {e}")

        return best_weights, best_reward