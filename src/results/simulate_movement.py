import pickle
import torch
import numpy as np
import pandas as pd
import configparser

from init_mlp import MLP
from kinetics.nanorobot_solver import NanorobotSolver

# Load configuration
config = configparser.ConfigParser()
config.read("configfile.ini", encoding="utf-8")

# Read nanorobot modeling parameters
MODEL_NAME = config["PATHS"]["robot_model"]
OUTPUT_PATH = config["PATHS"]["output_path"]
CONFIG_NAMES_STR = config["NANOROBOT_MODELING"]["configuration_names"]
EXP_DATA_PATH_A = config["NANOROBOT_MODELING"]["path_to_experimental_data_a"]
EXP_DATA_PATH_B = config["NANOROBOT_MODELING"]["path_to_experimental_data_b"]
SIM_TIME_STEP = float(config["NANOROBOT_MODELING"]["sim_time_step"])
SIM_TOTAL_TIME = float(config["NANOROBOT_MODELING"]["sim_total_time"])
INITIAL_CONFIG_IDX = int(config["NANOROBOT_MODELING"]["initial_configuration_idx"])

# Define parameters to be optimized (corresponding to MLP output)
PARAMETER_NAMES = [
    "kBT", "lp_s", "lc_s", "lc_d", "E_b", "E_b_azo_trans",
    "di_DNA", "n_D1", "n_D2", "n_S1", "n_gray", "n_hairpin_1", "n_hairpin_2",
    "n_azo_1", "n_azo_2", "n_T_hairpin_1", "n_T_hairpin_2", "n_track_1", "n_track_2",
    "k0", "k_mig", "drt_z", "drt_s", "dE_TYE"
]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = len(PARAMETER_NAMES)
    output_dim = len(PARAMETER_NAMES)

    # Initialize MLP model
    mlp_model = MLP(input_size=input_dim, output_size=output_dim, hidden_sizes=[50, 50], device=device)

    try:
        # Load trained weights
        with open(OUTPUT_PATH+"best_mlp_weights.pkl", "rb") as f:
            loaded_weights = pickle.load(f)
        mlp_model.set_weights(loaded_weights)
        print("Trained MLP model loaded successfully.")

        # Generate parameters using the loaded MLP model
        # For simulation, we can use a fixed noise input or generate a new one
        # Here, we'll use a random noise input to generate parameters
        noise_input = np.random.randn(1, mlp_model.input_size)
        generated_array = mlp_model.predict(noise_input)
        params = {name: float(value) for name, value in zip(PARAMETER_NAMES, generated_array[0])}
        print("Generated kinetic parameters:")
        for k, v in params.items():
            print(f"  {k}: {v:.4f}")

        # Initialize NanorobotSolver with the generated parameters
        nanorobot_solver = NanorobotSolver(MODEL_NAME, CONFIG_NAMES_STR, EXP_DATA_PATH_A, EXP_DATA_PATH_B)
        nanorobot_solver.set_parameters(params)

        # Define initial configuration distribution
        initial_P = np.zeros(nanorobot_solver.num_configs)
        initial_P[INITIAL_CONFIG_IDX] = 1.0

        # Define light schedule (None for now, as in main.py)
        light_schedule = None

        # Run kinetic simulation
        print("Running nanorobot movement simulation...")
        simulated_df = nanorobot_solver.simulate(initial_P, SIM_TIME_STEP, SIM_TOTAL_TIME, light_schedule)
        print("Simulation complete.")

        # Save simulated data to CSV
        output_csv_path = "results/compare_data/simulated_movement.csv"
        simulated_df.to_csv(output_csv_path, index=False)
        print(f"Simulated movement data saved to {output_csv_path}")

    except FileNotFoundError:
        print("Error: best_mlp_weights.pkl not found. Please ensure the training was completed.")
    except Exception as e:
        print(f"An error occurred during simulation: {e}")


