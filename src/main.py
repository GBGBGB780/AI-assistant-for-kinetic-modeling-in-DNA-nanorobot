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
MODEL_NAME = config["PATHS"]["robot_model"]
CONFIG_NAMES_STR = config["NANOROBOT_MODELING"]["configuration_names"]
EXP_DATA_PATH_A = config["NANOROBOT_MODELING"]["path_to_experimental_data_a"]
EXP_DATA_PATH_B = config["NANOROBOT_MODELING"]["path_to_experimental_data_b"]
SIM_TIME_STEP = float(config["NANOROBOT_MODELING"]["sim_time_step"])
SIM_TOTAL_TIME = float(config["NANOROBOT_MODELING"]["sim_total_time"])
INITIAL_CONFIG_IDX = int(config["NANOROBOT_MODELING"]["initial_configuration_idx"])
REWARD_FLAG = int(config["REWARDS"]["reward_flag"])

# 加载纳米机器人动力学求解器 (模拟与评估)
nanorobot_solver = NanorobotSolver(MODEL_NAME, CONFIG_NAMES_STR, EXP_DATA_PATH_A, EXP_DATA_PATH_B)

# 定义需要优化的参数名称列表（与MLP输出对应）,删除了'E_b_azo_cis'
PARAMETER_NAMES = [
    "kBT", "lp_s", "lc_s", "lc_d", "E_b", "E_b_azo_trans",
    "di_DNA", "n_D1", "n_D2", "n_S1", "n_gray", "n_hairpin_1", "n_hairpin_2",
    "n_azo_1", "n_azo_2", "n_T_hairpin_1", "n_T_hairpin_2", "n_track_1", "n_track_2",
    "k0", "k_mig", "drt_z", "drt_s", "dE_TYE"
]


def reward_func(weights, mlp_model):
    """
    奖励函数：将权重加载到MLP生成器，生成动力学参数，
    运行仿真并评估与实验数据的拟合度（负MSE）。
    """
    # 将权重设置到MLP模型
    mlp_model.set_weights(weights)

    rewards = []
    # 每次评估生成一组参数并模拟，重复num_evals次取平均
    num_evaluations = 1
    for _ in range(num_evaluations):
        # 生成参数：MLP以随机噪声为输入，输出一个参数数组
        # noise_input 应该是一个 PyTorch tensor
        noise_input = np.random.randn(1, mlp_model.input_size)
        generated_array = mlp_model.predict(noise_input)
        # 将输出数组映射为参数字典
        params = {name: float(value) for name, value in zip(PARAMETER_NAMES, generated_array[0])}

        # 设置纳米机器人模型的参数
        nanorobot_solver.set_parameters(params)

        # 定义初始构型分布（所有概率置零，选定一个初始态概率为1）
        initial_P = np.zeros(nanorobot_solver.num_configs)
        initial_P[INITIAL_CONFIG_IDX] = 1.0

        # 定义光照时间表（示例或从配置读取）
        light_schedule = None

        # 运行动力学模拟
        sim_df = nanorobot_solver.simulate(initial_P, SIM_TIME_STEP, SIM_TOTAL_TIME, light_schedule)

        # 评估与实验数据拟合的奖励（通常为负均方误差）
        reward = nanorobot_solver.evaluate_model(sim_df, REWARD_FLAG)
        rewards.append(reward)

    # 返回平均奖励
    return float(np.mean(rewards))


if __name__ == "__main__":
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化MLP生成器：输入和输出维度均为参数数量
    input_dim = len(PARAMETER_NAMES)
    output_dim = len(PARAMETER_NAMES)
    mlp_model = MLP(input_size=input_dim, output_size=output_dim, hidden_sizes=[50, 50], device=device)

    # 初始权重向量
    initial_weights = mlp_model.get_weights()

    # 从配置文件读取进化策略超参数（可选）
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

    # 创建并运行进化策略优化
    es = EvolutionStrategy(initial_weights, reward_func, mlp_model,
                           population_size=pop_size, sigma=sigma,
                           learning_rate=lr, num_iterations=n_iter)
    best_weights, best_reward = es.run()

    # 输出结果并保存最佳权重
    print(f"优化完成，最佳奖励 = {best_reward:.4f}")
    with open("best_mlp_weights.pkl", "wb") as f:
        pickle.dump(best_weights, f)
