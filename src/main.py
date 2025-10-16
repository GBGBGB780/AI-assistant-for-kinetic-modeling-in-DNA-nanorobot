# coding=gb2312
import numpy as np
import configparser
import pickle
import torch

from init_mlp import MLP
from evolution_strategy import EvolutionStrategy
from kinetics.nanorobot_solver import NanorobotSolver

# 加载配置文件
config = configparser.ConfigParser()
config.read("configfile.ini", encoding="utf-8")

# 读取纳米机器人建模相关参数
# 自动设置固定参数以及带训练参数
fixed_params = {}
trainable_params_names = []
all_physical_params = config['PHYSICAL_PARAMETERS']

for name, value in all_physical_params.items():
    if value.strip() == "":
        # 如果值为空，则为待训练参数
        trainable_params_names.append(name)
    else:
        # 如果有值，则为固定参数
        fixed_params[name] = float(value)

print(f"将要训练 {len(trainable_params_names)} 个参数: {trainable_params_names}")
print(f"将使用 {len(fixed_params)} 个固定参数。")

# 读取待训练参数的范围
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
# 加载纳米机器人动力学求解器 (模拟与评估)
nanorobot_solver = NanorobotSolver(fixed_params, EXP_DATA_PATH_A)


def reward_func(weights, mlp_model):
    """
    奖励函数：将权重加载到MLP生成器，生成动力学参数，
    运行仿真并评估与实验数据的拟合度（负MSE）。
    """
    # 将权重设置到MLP模型
    mlp_model.set_weights(weights)

    rewards = []
    # 每次评估生成一组参数并模拟，重复num_evals次取平均
    num_evaluations = 5
    for _ in range(num_evaluations):
        # 生成参数：MLP以随机噪声为输入，输出一个参数数组
        # noise_input 应该是一个 PyTorch tensor
        noise_input = np.random.randn(1, mlp_model.input_size)
        generated_array = mlp_model.predict(noise_input)
        trained_params = {name: float(value) for name, value in zip(trainable_params_names, generated_array[0])}

        # 合并固定参数和训练参数
        all_params = fixed_params.copy()
        all_params.update(trained_params)

        # 设置参数并运行模拟
        nanorobot_solver.set_parameters(all_params)

        light_schedule = []
        current_time = 0

        # 根据起始模式确定光照周期的顺序
        if LIGHT_START_MODE == 0:  # 先 vis 后 uv
            phases = [('visible', CYCLE_DURATION_VIS), ('uv', CYCLE_DURATION_UV)]
        else:  # 先 uv 后 vis (默认为1或其他任何值)
            phases = [('uv', CYCLE_DURATION_UV), ('visible', CYCLE_DURATION_VIS)]

        # 循环添加光照阶段，直到达到总模拟时间
        # 添加安全检查，防止因duration为0导致无限循环
        if sum(p[1] for p in phases) <= 0:
            print("Warning: Total cycle duration is not positive. No light schedule will be generated.")
        else:
            while current_time < SIM_TOTAL_TIME:
                for light_type, duration in phases:
                    if duration > 0:  # 仅当持续时间大于0时才添加
                        current_time += duration
                        light_schedule.append((current_time, light_type))

        # 定义初始构型
        initial_P = np.zeros(nanorobot_solver.num_configs)
        initial_P[INITIAL_CONFIG_IDX] = 1.0

        # 调用模拟函数
        sim_df = nanorobot_solver.run_simulation(initial_P, SIM_TOTAL_TIME, light_schedule)

        # 评估奖励
        reward = nanorobot_solver.evaluate_model(sim_df, REWARD_FLAG)
        rewards.append(reward)

    return float(np.mean(rewards))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化MLP，其输入输出维度由待训练参数的数量决定
    input_dim = len(trainable_params_names)
    output_dim = len(trainable_params_names)
    mlp_model = MLP(input_size=input_dim, output_size=output_dim, hidden_sizes=[50, 50],
                    device=device, param_names=trainable_params_names, param_ranges=param_ranges)

    # 初始权重向量
    initial_weights = mlp_model.get_weights()

    # 从配置文件读取进化策略超参数
    pop_size = int(config["EVOSTRAT"]["population_size"])
    sigma = float(config["EVOSTRAT"]["noise"])
    lr = float(config["EVOSTRAT"]["learning_rate"])
    n_iter = int(config["EVOSTRAT"]["generations"])

    # 创建并运行进化策略优化
    es = EvolutionStrategy(initial_weights, reward_func, mlp_model,
                           population_size=pop_size, sigma=sigma,
                           learning_rate=lr, num_iterations=n_iter)
    best_weights, best_reward = es.run()

    # 输出结果并保存最佳权重
    print(f"优化完成，最佳奖励 = {best_reward:.4f}")
    with open(OUTPUT_PATH+"best_mlp_weights.pkl", "wb") as f:
        pickle.dump(best_weights, f)
