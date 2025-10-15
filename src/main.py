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
OUTPUT_PATH = config["PATHS"]["output_path"]
EXP_DATA_PATH_A = config["NANOROBOT_MODELING"]["path_to_experimental_data_a"]
SIM_TOTAL_TIME = float(config["NANOROBOT_MODELING"]["sim_total_time"])
INITIAL_CONFIG_IDX = int(config["NANOROBOT_MODELING"]["initial_configuration_idx"])
REWARD_FLAG = int(config["REWARDS"]["reward_flag"])
MIN_PARAM = float(config["CONSTRAINTS"]["min_param"])
MAX_PARAM = float(config["CONSTRAINTS"]["max_param"])

CYCLE_DURATION_VIS = float(config["NANOROBOT_MODELING"]["cycle_duration_vis"])
CYCLE_DURATION_UV = float(config["NANOROBOT_MODELING"]["cycle_duration_uv"])
LIGHT_START_MODE = int(config["NANOROBOT_MODELING"]["light_start_mode"])
# �������׻����˶���ѧ����� (ģ��������)
nanorobot_solver = NanorobotSolver(MODEL_NAME, EXP_DATA_PATH_A)


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

        light_schedule = []
        current_time = 0

        # ������ʼģʽȷ���������ڵ�˳��
        if LIGHT_START_MODE == 0:  # �� vis �� uv
            phases = [('visible', CYCLE_DURATION_VIS), ('uv', CYCLE_DURATION_UV)]
        else:  # �� uv �� vis (Ĭ��Ϊ1�������κ�ֵ)
            phases = [('uv', CYCLE_DURATION_UV), ('visible', CYCLE_DURATION_VIS)]

        # ѭ����ӹ��ս׶Σ�ֱ���ﵽ��ģ��ʱ��
        # ��Ӱ�ȫ��飬��ֹ��durationΪ0��������ѭ��
        if sum(p[1] for p in phases) <= 0:
            print("Warning: Total cycle duration is not positive. No light schedule will be generated.")
        else:
            while current_time < SIM_TOTAL_TIME:
                for light_type, duration in phases:
                    if duration > 0:  # ��������ʱ�����0ʱ�����
                        current_time += duration
                        light_schedule.append((current_time, light_type))

        # �����ʼ����
        initial_P = np.zeros(nanorobot_solver.num_configs)
        initial_P[INITIAL_CONFIG_IDX] = 1.0

        # ����ģ�⺯��
        sim_df = nanorobot_solver.run_simulation(initial_P, SIM_TOTAL_TIME, light_schedule)

        # ��������
        reward = nanorobot_solver.evaluate_model(sim_df, REWARD_FLAG)
        rewards.append(reward)

    return float(np.mean(rewards))


if __name__ == "__main__":
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
    with open(OUTPUT_PATH+"best_mlp_weights.pkl", "wb") as f:
        pickle.dump(best_weights, f)
