import numpy as np
import pandas as pd


def load_iso_data(iso: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads ISO data (load, wind, solar, natural gas price)
    :param iso: ISO name
    """
    df = pd.read_csv(f'data/{iso.upper()}/{iso.lower()}.csv', index_col=0)
    x_cols = list(set(df.columns) - set(['PRICE']))
    X = df[x_cols].values
    y = df['PRICE'].values
    return X, y


def load_fuelmix_data(iso: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads fuel mix data (includes also load, price, and natural gas price)
    :param iso: ISO name
    """
    df = pd.read_csv(f'data/{iso.upper()}/fuelmix.csv', index_col=0)
    x_cols = list(set(df.columns) - set(['PRICE']))
    X = df[x_cols].values
    y = df['PRICE'].values
    return X, y
