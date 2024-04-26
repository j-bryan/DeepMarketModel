import torch


def find_accelerator() -> torch.device:
    """
    Find the accelerator hardware available. Check first for CUDA GPU, then MPS, then default to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
