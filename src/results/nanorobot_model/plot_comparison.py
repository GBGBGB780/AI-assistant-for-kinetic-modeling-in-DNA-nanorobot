import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read("configfile.ini", encoding="utf-8")

EXP_DATA_PATH_A = config["NANOROBOT_MODELING"]["path_to_experimental_data_a"]

if __name__ == "__main__":
    # Load simulated data
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
    plt.plot(experimental_df['Time'], experimental_df['FAM/FAM T (+)'], label='Experimental FAM', color='blue', alpha=0.7)
    plt.plot(simulated_df['Time'], sim_fam, label='Simulated FAM', color='red', linestyle='--', alpha=0.7)
    plt.title('FAM Signal Comparison')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True)

    # TYE/TYE T (-)
    plt.subplot(3, 1, 2)
    plt.plot(experimental_df['Time'], experimental_df['TYE/TYE T (-)'], label='Experimental TYE', color='blue', alpha=0.7)
    plt.plot(simulated_df['Time'], sim_tye, label='Simulated TYE', color='red', linestyle='--', alpha=0.7)
    plt.title('TYE Signal Comparison')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True)

    # CY5/CY5 T (m)
    plt.subplot(3, 1, 3)
    plt.plot(experimental_df['Time'], experimental_df['CY5/CY5 T (m)'], label='Experimental CY5', color='blue', alpha=0.7)
    plt.plot(simulated_df['Time'], sim_cy5, label='Simulated CY5', color='red', linestyle='--', alpha=0.7)
    plt.title('CY5 Signal Comparison')
    plt.ylabel('Signal')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('simulation_vs_experimental_data.png')
    print("Comparison plot saved as simulation_vs_experimental_data.png")


