"""
Loads the data in a pytorch dataloader
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_loading import load_iso_data as _load_iso_data
from .data_loading import load_fuelmix_data as _load_fuelmix_data


def _to_dataloader(data: torch.Tensor, batch_size: int = 32) -> torch.utils.data.DataLoader:
    """
    Load the data into a pytorch dataloader

    Args:
    data (torch.Tensor): The data to load
    batch_size (int): The batch size to use

    Returns:
    torch.utils.data.DataLoader: The dataloader
    """
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


def load_iso_data(iso: str, batch_size: int = 32) -> DataLoader:
    """
    Loads ISO data (load, wind, solar, natural gas price)
    :param iso: ISO name
    :param batch_size: The batch size to use
    :return dataloader: The dataloader
    """
    X, y = _load_iso_data(iso)
    y = y.reshape(-1, 1)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataloader = _to_dataloader(torch.utils.data.TensorDataset(X, y))
    return dataloader


def load_fuelmix_data(iso: str, batch_size: int = 32) -> DataLoader:
    """
    Loads fuel mix data (includes also load, price, and natural gas price)
    :param iso: ISO name
    :param batch_size: The batch size to use
    :return dataloader: The dataloader
    """
    X, y = _load_fuelmix_data(iso)
    y = y.reshape(-1, 1)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataloader = _to_dataloader(torch.utils.data.TensorDataset(X, y))
    return dataloader
