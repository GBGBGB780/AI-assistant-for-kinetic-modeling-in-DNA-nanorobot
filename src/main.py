# coding=gb2312
import numpy as np
import configparser
import pickle
import torch
import multiprocessing
import traceback

from init_mlp import MLP
from evolution_strategy import EvolutionStrategy
from kinetics.nanorobot_solver import NanorobotSolver

# --- 1. ���������ļ� (�ޱ仯) ---
config = configparser.ConfigParser()
config.read("configfile.ini", encoding="utf-8")

# --- 2. ����ʶ��̶������ʹ�ѵ������ (�ޱ仯) ---
fixed_params = {}
trainable_params_names = []
all_physical_params = config['PHYSICAL_PARAMETERS']
for name, value in all_physical_params.items():
    if value.strip() == "":
        trainable_params_names.append(name)
    else:
        fixed_params[name] = float(value)
print(f"��Ҫѵ�� {len(trainable_params_names)} ������: {trainable_params_names}")
print(f"��ʹ�� {len(fixed_params)} ���̶�������")

# --- 3. ��ȡ�������� (�ޱ仯) ---
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

# ����ȫ�ֱ������������ӽ����г�ʼ������
g_mlp_model = None
g_nanorobot_solver = None


# ���幤�����̵ĳ�ʼ������
def init_worker(mlp_config, solver_config):
    """
    �˺�����ÿ��������������ʱ������һ�Ρ�
    ��������ÿ�������д���������MLP��Solverʵ����
    """
    global g_mlp_model, g_nanorobot_solver
    g_mlp_model = MLP(**mlp_config)
    g_nanorobot_solver = NanorobotSolver(**solver_config)
    print(f"�������� {multiprocessing.current_process().pid} �ѳ�ʼ����")


# ���嵥����ѡ�ߵ����������������ӽ�����ִ�У�
def evaluate_candidate_reward(weights):
    """
    ����ʵ����ÿ��CPU���������еĹ���������
    ��ֻ����һ��������weights��
    """
    global g_mlp_model, g_nanorobot_solver

    pid = multiprocessing.current_process().pid

    rewards = []
    num_evaluations = 2  # ������������������ٶ�

    # ����ģ��-���������������try...except����
    try:
        for i in range(num_evaluations):
            g_mlp_model.set_weights(weights)
            noise_input = np.random.randn(1, g_mlp_model.input_size)
            generated_array = g_mlp_model.predict(noise_input)
            trained_params = {name: float(value) for name, value in zip(trainable_params_names, generated_array[0])}

            all_params = fixed_params.copy()
            all_params.update(trained_params)
            g_nanorobot_solver.set_parameters(all_params)

            light_schedule = []
            current_time = 0
            phases = [('visible', CYCLE_DURATION_VIS), ('uv', CYCLE_DURATION_UV)] if LIGHT_START_MODE == 0 else [
                ('uv', CYCLE_DURATION_UV), ('visible', CYCLE_DURATION_VIS)]

            if sum(p[1] for p in phases) > 0:
                while current_time < SIM_TOTAL_TIME:
                    for light_type, duration in phases:
                        if duration > 0:
                            current_time += duration
                            light_schedule.append((current_time, light_type))

            initial_P = np.zeros(g_nanorobot_solver.num_configs)
            initial_P[INITIAL_CONFIG_IDX] = 1.0
            sim_df = g_nanorobot_solver.run_simulation(initial_P, SIM_TOTAL_TIME, light_schedule)

            reward = g_nanorobot_solver.evaluate_model(sim_df, REWARD_FLAG)
            rewards.append(reward)

        # ����ƽ������
        avg_reward_numpy = np.mean(rewards)
        final_reward = float(avg_reward_numpy)
        return final_reward

    except Exception as e:
        # ��������ǿ
        print(f"\n[��������PID: {pid}] ��������: {e}", flush=True)
        traceback.print_exc()  # ��ӡ��ϸ�����ջ
        print(f"[��������PID: {pid}] ����ʧ�ܣ����سͷ�ֵ -1000.0", flush=True)
        return -1000.0  # ����һ����ʾʧ�ܵġ��ܲ�Ľ��


# --- ��������� ---
if __name__ == "__main__":
    # ǿ����CPU�����У���Ϊģ����Ҫ���ӽ������ؽ��������ȫ������ݵĲ��л���ʽ
    device = torch.device("cpu")
    print(f"Using device: {device} for parallel processing.")

    # ׼�����ݸ��ӽ��̳�ʼ�������������ֵ�
    mlp_config = {
        "input_size": len(trainable_params_names),
        "output_size": len(trainable_params_names),
        "hidden_sizes": [50, 50],
        "device": device,
        "param_names": trainable_params_names,
        "param_ranges": param_ranges
    }
    solver_config = {
        "initial_parameters": fixed_params,
        "experimental_data_path_a": EXP_DATA_PATH_A
    }

    # ������������̳أ�����HPC����֧�֣�
    try:
        import os

        if 'PBS_NCPUS' in os.environ:
            num_workers = int(os.environ.get('PBS_NCPUS'))
            print(f"��⵽PBS������ʹ�÷���� {num_workers} ��CPU���ġ�")
        elif 'SLURM_CPUS_PER_TASK' in os.environ:
            num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK'))
            print(f"��⵽SLURM������ʹ�÷���� {num_workers} ��CPU���ġ�")
        else:
            num_workers = int(config["EVOSTRAT"].get("n_threads", multiprocessing.cpu_count()))
            print(f"δ��⵽HPC������ʹ�� {num_workers} ��CPU���ġ�")
    except (ImportError, ValueError, TypeError):
        num_workers = int(config["EVOSTRAT"].get("n_threads", multiprocessing.cpu_count()))
        print(f"HPC�������ʧ�ܣ�ʹ�� {num_workers} ��CPU���ġ�")

    print(f"���� {num_workers} ����������...")

    with multiprocessing.Pool(processes=num_workers, initializer=init_worker,
                              initargs=(mlp_config, solver_config)) as pool:
        # �����������д���һ����ʱMLP����ȡ��ʼȨ��
        temp_mlp_for_weights = MLP(**mlp_config)
        initial_weights = temp_mlp_for_weights.get_weights()

        pop_size = int(config["EVOSTRAT"]["population_size"])
        sigma = float(config["EVOSTRAT"]["noise"])
        lr = float(config["EVOSTRAT"]["learning_rate"])
        n_iter = int(config["EVOSTRAT"]["generations"])

        # �����̳غ͹����������ݸ����������㷨
        es = EvolutionStrategy(initial_weights=initial_weights,
                               reward_function=evaluate_candidate_reward,
                               pool=pool,
                               population_size=pop_size, sigma=sigma,
                               learning_rate=lr, num_iterations=n_iter)

        best_weights, best_reward = es.run()

        # ���������������Ȩ��
        print(f"�Ż���ɣ���ѽ��� = {best_reward:.4f}")
        with open(OUTPUT_PATH + "best_mlp_weights.pkl", "wb") as f:
            pickle.dump(best_weights, f)
