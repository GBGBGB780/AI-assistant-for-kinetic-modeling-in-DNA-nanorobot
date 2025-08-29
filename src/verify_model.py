import pickle
import torch
import numpy as np
from init_mlp import MLP
import configparser
import matplotlib.pyplot as plt
import pandas as pd

# Load configuration
config = configparser.ConfigParser()
config.read("configfile.ini", encoding="utf-8")
EXP_DATA_PATH_A = config["NANOROBOT_MODELING"]["path_to_experimental_data_a"]

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
    mlp_model = MLP(input_size=input_dim, output_size=output_dim, hidden_sizes=[50, 50], device=device)

    try:
        with open("best_mlp_weights.pkl", "rb") as f:
            loaded_weights = pickle.load(f)
        print("Model weights loaded successfully.")

        # Set the loaded weights to a new MLP model instance
        loaded_mlp_model = MLP(input_size=input_dim, output_size=output_dim, hidden_sizes=[50, 50], device=device)
        loaded_mlp_model.set_weights(loaded_weights)
        print("Weights successfully set to a new MLP model instance.")

        # Verify by making a prediction
        dummy_input = np.random.randn(1, input_dim)
        output = loaded_mlp_model.predict(dummy_input)
        print(f"Dummy prediction output shape: {output.shape}")
        print("Model verification successful.")

    except FileNotFoundError:
        print("Error: best_mlp_weights.pkl not found. Please run main.py first to train the model.")
    except Exception as e:
        print(f"An error occurred during model loading or verification: {e}")

        # make a compare graph
        try:
            simulated_df = pd.read_csv("simulated_movement.csv")
            print("Simulated data loaded successfully.")
        except FileNotFoundError:
            print("Error: simulated_movement.csv not found. Please run simulate_movement.py first.")
            exit()

        # Load experimental data
        try:
            experimental_df = pd.read_csv(EXP_DATA_PATH_A)
            print("Experimental data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: {EXP_DATA_PATH_A} not found. Please provide the correct path.")
            exit()

        # Ensure 'Time' column exists in both dataframes
        if 'Time' not in simulated_df.columns or 'Time' not in experimental_df.columns:
            print("Error: 'Time' column not found in one or both dataframes.")
            exit()

        # Define the columns to plot (based on nanorobot_solver.py's evaluate_model)
        # These correspond to FAM, TYE, CY5 signals
        # Simulated data has P_0 to P_7
        # Experimental data has 'FAM/FAM T (+)', 'TYE/TYE T (-)', 'CY5/CY5 T (m)'

        # Calculate combined simulated signals
        sim_fam = simulated_df['P_0'] + simulated_df['P_1']
        sim_tye = simulated_df['P_2'] + simulated_df['P_3']
        sim_cy5 = simulated_df['P_4'] + simulated_df['P_5'] + simulated_df['P_6'] + simulated_df['P_7']

        # Plotting
        plt.figure(figsize=(12, 8))

        # FAM/FAM T (+)
        plt.subplot(3, 1, 1)
        plt.plot(experimental_df['Time'], experimental_df['FAM/FAM T (+)'], label='Experimental FAM', color='blue',
                 alpha=0.7)
        plt.plot(simulated_df['Time'], sim_fam, label='Simulated FAM', color='red', linestyle='--', alpha=0.7)
        plt.title('FAM Signal Comparison')
        plt.ylabel('Signal')
        plt.legend()
        plt.grid(True)

        # TYE/TYE T (-)
        plt.subplot(3, 1, 2)
        plt.plot(experimental_df['Time'], experimental_df['TYE/TYE T (-)'], label='Experimental TYE', color='blue',
                 alpha=0.7)
        plt.plot(simulated_df['Time'], sim_tye, label='Simulated TYE', color='red', linestyle='--', alpha=0.7)
        plt.title('TYE Signal Comparison')
        plt.ylabel('Signal')
        plt.legend()
        plt.grid(True)

        # CY5/CY5 T (m)
        plt.subplot(3, 1, 3)
        plt.plot(experimental_df['Time'], experimental_df['CY5/CY5 T (m)'], label='Experimental CY5', color='blue',
                 alpha=0.7)
        plt.plot(simulated_df['Time'], sim_cy5, label='Simulated CY5', color='red', linestyle='--', alpha=0.7)
        plt.title('CY5 Signal Comparison')
        plt.ylabel('Signal')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('simulation_vs_experimental_data.png')
        print("Comparison plot saved as simulation_vs_experimental_data.png")


