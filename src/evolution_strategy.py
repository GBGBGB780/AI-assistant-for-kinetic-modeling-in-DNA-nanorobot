# coding=gb2312
import numpy as np


class EvolutionStrategy:
    """
    ��Ȼ�������� (NES) �Ż��㷨��ʹ�� multiprocessing.Pool ���в���������
    """

    # __init__ ���ڽ���һ�� 'pool' �����Ƴ��� mlp_model
    def __init__(self, initial_weights, reward_function, pool,
                 population_size=50, sigma=0.1, learning_rate=0.05, num_iterations=100):
        self.weights = initial_weights.copy()
        self.reward_function = reward_function  # ���� main.py �еĶ�����������
        self.pool = pool  # �洢���̳�
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

            # ʹ�ý��̳ز����������к�ѡ��
            # ��ԭ����ʱ�� for ѭ���滻Ϊһ�� pool.map ����
            print(
                f"\n--- [������] ��ʼ�� {iteration + 1}/{self.num_iterations} ������ (�ַ� {self.pop_size} ������)... ---",
                flush=True)
            rewards = np.array(self.pool.map(self.reward_function, candidates))

            print(f"--- [������] �� {iteration + 1} ��������ɡ���ʼ����Ȩ��... ---", flush=True)

            # ���� best
            max_idx = int(np.argmax(rewards))
            if rewards[max_idx] > best_reward and np.isfinite(rewards[max_idx]):
                best_reward = float(rewards[max_idx])
                # ֱ�Ӵ� candidates �����л�ȡ���Ȩ��
                best_weights = candidates[max_idx].copy()

            # �� rewards ���Ƚ���׼��
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

            # ��������ݶȣ���������
            weighted_noises = np.dot(norm_rewards, noises)
            grad = weighted_noises / (self.pop_size * self.sigma)

            # ����Ȩ��
            self.weights = self.weights + self.lr * grad

            # ��ӡ����
            print("���� {}/{}����ѽ��� = {:.4f}����ǰƽ������ = {:.4f}".format(
                iteration + 1, self.num_iterations, float(best_reward if np.isfinite(best_reward) else -np.inf),
                mean
            ))

        return best_weights, best_reward
