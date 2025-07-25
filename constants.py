import torch 

DEFAULT_MEMORY_DIMENSION = 32
DEFAULT_TRAJECTORY_LENGTH = 256
DEFAULT_MEMORY_ENTRIES = 100

DEFAULT_MOTIF_LEN = 4

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')