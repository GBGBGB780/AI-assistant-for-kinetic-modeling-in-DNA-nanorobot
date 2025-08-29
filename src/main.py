# coding=gb2312
import numpy as np
import configparser
import pickle
import torch

from init_mlp import MLP
from evolution_strategy import EvolutionStrategy
from kinetics.nanorobot_solver import NanorobotSolver

# ���������ļ�
config = configparser.ConfigParser()
config.read("configfile.ini", encoding="utf-8")

# ��ȡ���׻����˽�ģ��ز���
MODEL_NAME = config["PATHS"]["robot_model"]
CONFIG_NAMES_STR = config["NANOROBOT_MODELING"]["configuration_names"]
EXP_DATA_PATH_A = config["NANOROBOT_MODELING"]["path_to_experimental_data_a"]
EXP_DATA_PATH_B = config["NANOROBOT_MODELING"]["path_to_experimental_data_b"]
SIM_TIME_STEP = float(config["NANOROBOT_MODELING"]["sim_time_step"])
SIM_TOTAL_TIME = float(config["NANOROBOT_MODELING"]["sim_total_time"])
INITIAL_CONFIG_IDX = int(config["NANOROBOT_MODELING"]["initial_configuration_idx"])
REWARD_FLAG = int(config["REWARDS"]["reward_flag"])

# �������׻����˶���ѧ����� (ģ��������)
nanorobot_solver = NanorobotSolver(MODEL_NAME, CONFIG_NAMES_STR, EXP_DATA_PATH_A, EXP_DATA_PATH_B)

# ������Ҫ�Ż��Ĳ��������б���MLP�����Ӧ��,ɾ����'E_b_azo_cis'
PARAMETER_NAMES = [
    "kBT", "lp_s", "lc_s", "lc_d", "E_b", "E_b_azo_trans",
    "di_DNA", "n_D1", "n_D2", "n_S1", "n_gray", "n_hairpin_1", "n_hairpin_2",
    "n_azo_1", "n_azo_2", "n_T_hairpin_1", "n_T_hairpin_2", "n_track_1", "n_track_2",
    "k0", "k_mig", "drt_z", "drt_s", "dE_TYE"
]


def reward_func(weights, mlp_model):
    """
    ������������Ȩ�ؼ��ص�MLP�����������ɶ���ѧ������
    ���з��沢������ʵ�����ݵ���϶ȣ���MSE����
    """
    # ��Ȩ�����õ�MLPģ��
    mlp_model.set_weights(weights)

    rewards = []
    # ÿ����������һ�������ģ�⣬�ظ�num_evals��ȡƽ��
    num_evaluations = 1
    for _ in range(num_evaluations):
        # ���ɲ�����MLP���������Ϊ���룬���һ����������
        # noise_input Ӧ����һ�� PyTorch tensor
        noise_input = np.random.randn(1, mlp_model.input_size)
        generated_array = mlp_model.predict(noise_input)
        # ���������ӳ��Ϊ�����ֵ�
        params = {name: float(value) for name, value in zip(PARAMETER_NAMES, generated_array[0])}

        # �������׻�����ģ�͵Ĳ���
        nanorobot_solver.set_parameters(params)

        # �����ʼ���ͷֲ������и������㣬ѡ��һ����ʼ̬����Ϊ1��
        initial_P = np.zeros(nanorobot_solver.num_configs)
        initial_P[INITIAL_CONFIG_IDX] = 1.0

        # �������ʱ���ʾ��������ö�ȡ��
        light_schedule = None

        # ���ж���ѧģ��
        sim_df = nanorobot_solver.simulate(initial_P, SIM_TIME_STEP, SIM_TOTAL_TIME, light_schedule)

        # ������ʵ��������ϵĽ�����ͨ��Ϊ��������
        reward = nanorobot_solver.evaluate_model(sim_df, REWARD_FLAG)
        rewards.append(reward)

    # ����ƽ������
    return float(np.mean(rewards))


if __name__ == "__main__":
    # ȷ���豸
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ��ʼ��MLP����������������ά�Ⱦ�Ϊ��������
    input_dim = len(PARAMETER_NAMES)
    output_dim = len(PARAMETER_NAMES)
    mlp_model = MLP(input_size=input_dim, output_size=output_dim, hidden_sizes=[50, 50], device=device)

    # ��ʼȨ������
    initial_weights = mlp_model.get_weights()

    # �������ļ���ȡ�������Գ���������ѡ��
    try:
        pop_size = int(config["EVOLUTION"]["population_size"])
        sigma = float(config["EVOLUTION"]["sigma"])
        lr = float(config["EVOLUTION"]["learning_rate"])
        n_iter = int(config["EVOLUTION"]["generations"])
    except KeyError:
        pop_size = 50
        sigma = 0.1
        lr = 0.01
        n_iter = 1000

    # ���������н��������Ż�
    es = EvolutionStrategy(initial_weights, reward_func, mlp_model,
                           population_size=pop_size, sigma=sigma,
                           learning_rate=lr, num_iterations=n_iter)
    best_weights, best_reward = es.run()

    # ���������������Ȩ��
    print(f"�Ż���ɣ���ѽ��� = {best_reward:.4f}")
    with open("best_mlp_weights.pkl", "wb") as f:
        pickle.dump(best_weights, f)
