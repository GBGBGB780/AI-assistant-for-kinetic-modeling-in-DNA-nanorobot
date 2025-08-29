import pickle
import torch
import numpy as np
from init_mlp import MLP
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read("configfile.ini", encoding="utf-8")

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


