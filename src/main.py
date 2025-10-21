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

# --- 1. 加载配置文件 (无变化) ---
config = configparser.ConfigParser()
config.read("configfile.ini", encoding="utf-8")

# --- 2. 智能识别固定参数和待训练参数 (无变化) ---
fixed_params = {}
trainable_params_names = []
all_physical_params = config['PHYSICAL_PARAMETERS']
for name, value in all_physical_params.items():
    if value.strip() == "":
        trainable_params_names.append(name)
    else:
        fixed_params[name] = float(value)
print(f"将要训练 {len(trainable_params_names)} 个参数: {trainable_params_names}")
print(f"将使用 {len(fixed_params)} 个固定参数。")

# --- 3. 读取其他配置 (无变化) ---
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

# 定义全局变量，用于在子进程中初始化对象
g_mlp_model = None
g_nanorobot_solver = None


# 定义工作进程的初始化函数
def init_worker(mlp_config, solver_config):
    """
    此函数在每个工作进程启动时被调用一次。
    它负责在每个进程中创建独立的MLP和Solver实例。
    """
    global g_mlp_model, g_nanorobot_solver
    g_mlp_model = MLP(**mlp_config)
    g_nanorobot_solver = NanorobotSolver(**solver_config)
    print(f"工作进程 {multiprocessing.current_process().pid} 已初始化。")


# 定义单个候选者的评估函数（将在子进程中执行）
def evaluate_candidate_reward(weights):
    """
    这是实际在每个CPU核心上运行的工作函数。
    它只接收一个参数：weights。
    """
    global g_mlp_model, g_nanorobot_solver

    pid = multiprocessing.current_process().pid

    rewards = []
    num_evaluations = 2  # 减少评估次数以提高速度

    # 包裹模型-求解器评估过程在try...except块中
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

        # 返回平均奖励
        avg_reward_numpy = np.mean(rewards)
        final_reward = float(avg_reward_numpy)
        return final_reward

    except Exception as e:
        # 错误处理增强
        print(f"\n[工作进程PID: {pid}] 发生错误: {e}", flush=True)
        traceback.print_exc()  # 打印详细错误堆栈
        print(f"[工作进程PID: {pid}] 评估失败，返回惩罚值 -1000.0", flush=True)
        return -1000.0  # 返回一个表示失败的、很差的结果


# --- 主程序入口 ---
if __name__ == "__main__":
    # 强制在CPU上运行，因为模型需要在子进程中重建，这是最安全、最兼容的并行化方式
    device = torch.device("cpu")
    print(f"Using device: {device} for parallel processing.")

    # 准备传递给子进程初始化函数的配置字典
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

    # 创建并管理进程池（增加HPC环境支持）
    try:
        import os

        if 'PBS_NCPUS' in os.environ:
            num_workers = int(os.environ.get('PBS_NCPUS'))
            print(f"检测到PBS环境，使用分配的 {num_workers} 个CPU核心。")
        elif 'SLURM_CPUS_PER_TASK' in os.environ:
            num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK'))
            print(f"检测到SLURM环境，使用分配的 {num_workers} 个CPU核心。")
        else:
            num_workers = int(config["EVOSTRAT"].get("n_threads", multiprocessing.cpu_count()))
            print(f"未检测到HPC环境，使用 {num_workers} 个CPU核心。")
    except (ImportError, ValueError, TypeError):
        num_workers = int(config["EVOSTRAT"].get("n_threads", multiprocessing.cpu_count()))
        print(f"HPC环境检测失败，使用 {num_workers} 个CPU核心。")

    print(f"启动 {num_workers} 个工作进程...")

    with multiprocessing.Pool(processes=num_workers, initializer=init_worker,
                              initargs=(mlp_config, solver_config)) as pool:
        # 仅在主进程中创建一个临时MLP来获取初始权重
        temp_mlp_for_weights = MLP(**mlp_config)
        initial_weights = temp_mlp_for_weights.get_weights()

        pop_size = int(config["EVOSTRAT"]["population_size"])
        sigma = float(config["EVOSTRAT"]["noise"])
        lr = float(config["EVOSTRAT"]["learning_rate"])
        n_iter = int(config["EVOSTRAT"]["generations"])

        # 将进程池和工作函数传递给进化策略算法
        es = EvolutionStrategy(initial_weights=initial_weights,
                               reward_function=evaluate_candidate_reward,
                               pool=pool,
                               population_size=pop_size, sigma=sigma,
                               learning_rate=lr, num_iterations=n_iter)

        best_weights, best_reward = es.run()

        # 输出结果并保存最佳权重
        print(f"优化完成，最佳奖励 = {best_reward:.4f}")
        with open(OUTPUT_PATH + "best_mlp_weights.pkl", "wb") as f:
            pickle.dump(best_weights, f)
