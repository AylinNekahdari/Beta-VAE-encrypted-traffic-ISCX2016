import torch

SEED = 42
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
BETA = 4.0
LATENT_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/vpn_data/consolidated_traffic_data.csv"
MODEL_DIR = "model_outputs"