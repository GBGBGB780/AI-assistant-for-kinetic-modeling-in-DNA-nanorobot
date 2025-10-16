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
# �Զ����ù̶������Լ���ѵ������
fixed_params = {}
trainable_params_names = []
all_physical_params = config['PHYSICAL_PARAMETERS']

for name, value in all_physical_params.items():
    if value.strip() == "":
        # ���ֵΪ�գ���Ϊ��ѵ������
        trainable_params_names.append(name)
    else:
        # �����ֵ����Ϊ�̶�����
        fixed_params[name] = float(value)

print(f"��Ҫѵ�� {len(trainable_params_names)} ������: {trainable_params_names}")
print(f"��ʹ�� {len(fixed_params)} ���̶�������")

# ��ȡ��ѵ�������ķ�Χ
param_ranges = {}
if config.has_section('TRAINING_PARAMETER_RANGES'):
    for name, value in config.items('TRAINING_PARAMETER_RANGES'):
        min_val, max_val = [float(x.strip()) for x in value.split(',')]
        param_ranges[name] = (min_val, max_val)

OUTPUT_PATH = config["PATHS"]["output_path"]
EXP_DATA_PATH_A = config["NANOROBOT_MODELING"]["path_to_experimental_data_a"]
SIM_TOTAL_TIME = float(config["NANOROBOT_MODELING"]["sim_total_time"])
INITIAL_CONFIG_IDX = int(config["NANOROBOT_MODELING"]["initial_configuration_idx"])
REWARD_FLAG = int(config["REWARDS"]["reward_flag"])

CYCLE_DURATION_VIS = float(config["NANOROBOT_MODELING"]["cycle_duration_vis"])
CYCLE_DURATION_UV = float(config["NANOROBOT_MODELING"]["cycle_duration_uv"])
LIGHT_START_MODE = int(config["NANOROBOT_MODELING"]["light_start_mode"])
# �������׻����˶���ѧ����� (ģ��������)
nanorobot_solver = NanorobotSolver(fixed_params, EXP_DATA_PATH_A)


def reward_func(weights, mlp_model):
    """
    ������������Ȩ�ؼ��ص�MLP�����������ɶ���ѧ������
    ���з��沢������ʵ�����ݵ���϶ȣ���MSE����
    """
    # ��Ȩ�����õ�MLPģ��
    mlp_model.set_weights(weights)

    rewards = []
    # ÿ����������һ�������ģ�⣬�ظ�num_evals��ȡƽ��
    num_evaluations = 5
    for _ in range(num_evaluations):
        # ���ɲ�����MLP���������Ϊ���룬���һ����������
        # noise_input Ӧ����һ�� PyTorch tensor
        noise_input = np.random.randn(1, mlp_model.input_size)
        generated_array = mlp_model.predict(noise_input)
        trained_params = {name: float(value) for name, value in zip(trainable_params_names, generated_array[0])}

        # �ϲ��̶�������ѵ������
        all_params = fixed_params.copy()
        all_params.update(trained_params)

        # ���ò���������ģ��
        nanorobot_solver.set_parameters(all_params)

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

    # ��ʼ��MLP�����������ά���ɴ�ѵ����������������
    input_dim = len(trainable_params_names)
    output_dim = len(trainable_params_names)
    mlp_model = MLP(input_size=input_dim, output_size=output_dim, hidden_sizes=[50, 50],
                    device=device, param_names=trainable_params_names, param_ranges=param_ranges)

    # ��ʼȨ������
    initial_weights = mlp_model.get_weights()

    # �������ļ���ȡ�������Գ�����
    pop_size = int(config["EVOSTRAT"]["population_size"])
    sigma = float(config["EVOSTRAT"]["noise"])
    lr = float(config["EVOSTRAT"]["learning_rate"])
    n_iter = int(config["EVOSTRAT"]["generations"])

    # ���������н��������Ż�
    es = EvolutionStrategy(initial_weights, reward_func, mlp_model,
                           population_size=pop_size, sigma=sigma,
                           learning_rate=lr, num_iterations=n_iter)
    best_weights, best_reward = es.run()

    # ���������������Ȩ��
    print(f"�Ż���ɣ���ѽ��� = {best_reward:.4f}")
    with open(OUTPUT_PATH+"best_mlp_weights.pkl", "wb") as f:
        pickle.dump(best_weights, f)
