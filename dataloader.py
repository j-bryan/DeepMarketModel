import os
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, FunctionTransformer, QuantileTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer


import torch
from torch.utils.data import DataLoader, TensorDataset


class InvertibleColumnTransformer(ColumnTransformer):
    """
    Adds an inverse transform method to the standard sklearn.compose.ColumnTransformer.

    Warning this is flaky and use at your own risk.  Validation checks that the column count in
    `transformers` are in your object `X` to be inverted.  Reordering of columns will break things!

    Yanked from https://github.com/scikit-learn/scikit-learn/issues/11463
    """
    def inverse_transform(self, X):
        if isinstance(X,pd.DataFrame):
            X = X.to_numpy()

        arrays = []
        for name, indices in self.output_indices_.items():
            transformer = self.named_transformers_.get(name, None)
            arr = X[:, indices.start: indices.stop]

            if transformer in (None, "passthrough", "drop"):
                pass

            else:
                arr = transformer.inverse_transform(arr)

            arrays.append(arr)

        retarr = np.concatenate(arrays, axis=1)

        if retarr.shape[1] != X.shape[1]:
            raise ValueError(f"Received {X.shape[1]} columns but transformer expected {retarr.shape[1]}")

        return retarr


class BatchedTransformer:
    """
    Wrapper for a scikit-learn transformer that works with batched 2D and 3D torch.Tensor objects
    and numpy arrays.
    """
    def __init__(self, transformer: TransformerMixin | Pipeline):
        self.transformer = transformer
        # self.original_shape = None  # original shape of the tensor in the input space
        self.fitted = False
        self.n_vars = None

    def _validate_input(self, X: torch.Tensor | np.ndarray):
        if not self.fitted:
            self.n_vars = X.shape[-1]

        if not isinstance(X, (torch.Tensor, np.ndarray)):
            raise ValueError("X must be a torch.Tensor or numpy array.")

        if self.fitted and X.shape[-1] != self.n_vars:
            raise ValueError("X has a different number of features than the transformer was fitted on.")

    def _to_flat_np(self, tensor: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        Takes a tensor of shape (batch, seq, feature) or (batch, feature) and returns a numpy array
        of shape (batch * seq, feature) or (batch, feature) respectively.
        """
        if isinstance(tensor, torch.Tensor):
            x_np = tensor.detach().numpy()
        else:
            x_np = tensor
        x_np_flat = np.vstack(x_np)  # stacks the segments along the batch dimension; has no effect if already 2D
        return x_np_flat

    def _to_batched(self, ar: np.ndarray, batch_size: int, to_tensor: bool) -> torch.Tensor | np.ndarray:
        """
        Takes a numpy array of shape (batch * seq, feature) and returns a tensor of shape (batch, seq, feature)
        """
        # We need the batch size and the number of features to remain the same, but the sequence length
        # can change.
        last_dim_is_one = ar.shape[-1] == 1
        split_ar = np.squeeze(np.split(ar, batch_size, axis=0))
        tensor = torch.Tensor(split_ar) if to_tensor else split_ar
        if last_dim_is_one:  # if the last dimension was 1, we actually want to keep it that way
            tensor = tensor.unsqueeze(-1)
        return tensor

    def fit(self, X: torch.Tensor | np.ndarray) -> 'BatchedTransformer':
        self._validate_input(X)

        X_np = self._to_flat_np(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        self.transformer.fit(X_np)

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        self._validate_input(X)
        dim1 = X.shape[0]

        X_np = self._to_flat_np(X)
        X_transformed = self.transformer.transform(X_np)
        X_transformed = self._to_batched(X_transformed, dim1, isinstance(X, torch.Tensor))

        return X_transformed

    def inverse_transform(self, Xt: torch.Tensor) -> torch.Tensor:
        to_tensor = isinstance(Xt, torch.Tensor)
        self._validate_input(Xt)
        dim1 = Xt.shape[0]

        Xt_np = self._to_flat_np(Xt)
        Xt_transformed = self.transformer.inverse_transform(Xt_np)
        Xt_transformed = self._to_batched(Xt_transformed, dim1, to_tensor)

        return Xt_transformed

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)


def approx_capacities(iso: str) -> np.ndarray:
    """
    Approximates the effective capacities of the generators in the ISO. We assume that the effective
    capacities are equivalent to the maximum generator output in the ISO within some time period.
    We'll use monthly maximums, cast to hourly values.
    """
    # NOTE CAISO data isn't available all the way back to 2022-01-01! We'll need to adjust the start
    # date of the other data to match.
    possible_files = [f"data/{iso.upper()}/{iso.lower()}_generation.csv", f"data/{iso.upper()}/fuelmix.csv"]
    for file in possible_files:
        if not os.path.exists(file):
            continue
        data = pd.read_csv(file, index_col=0)
        break
    else:
        raise FileNotFoundError(f"No valid data file found for {iso}!")
    dates = pd.date_range(end=f"2022-12-31 23:00", periods=len(data), freq='h')
    data.index = dates
    monthly_maxima = []
    for (month, year), df in data.groupby([data.index.month, data.index.year]):
        max_gen = df.max(axis=0)
        monthly_maxima.append(np.tile(max_gen.to_numpy(), (len(df), 1)))
    return np.vstack(monthly_maxima)


def get_dataloaders(iso: str,
                    segment_length: int | None = None,
                    x_pipeline: BatchedTransformer | None = None,
                    y_pipeline: BatchedTransformer | None = None,
                    **kwargs):
    """
    Fetch the ISO data from file and return the dataloaders. Returns the dataloaders and the fitted
    pipelins (if any). The pipelines are returned in the order x_pipeline, y_pipeline.

    :param iso: str: The ISO to fetch data for
    :param segment_length: int: The length of the segment to use. Any leftover data will be discarded!
    :param x_pipeline: TensorTransformer: A pipeline to transform the X data. Works with tensors.
    :param y_pipeline: TensorTransformer: A pipeline to transform the y data. Works with tensors.
    :param kwargs: dict: Additional parameters handled (test_size, shuffle, random_state, batch_size)
    """
    # Fetch the data from file
    data = pd.read_csv(f"data/{iso.upper()}/{iso.lower()}.csv", index_col=0)
    y = data.pop("PRICE").to_numpy()
    X = data.to_numpy()

    if kwargs.pop("include_capacities", False):
        capacities = approx_capacities(iso)
        if len(capacities) != len(X):
            print(f"Length of capacities does not match length of data for {iso}! ({len(capacities)} vs {len(X)})")
        X = np.hstack((X[-len(capacities):], capacities))
        y = y[-len(capacities):]

    # If segment_length is not None, segment the data
    if segment_length is not None:
        num_segments = len(X) // segment_length
        X = X[:segment_length*num_segments].reshape(num_segments, segment_length, -1)
        y = y[:segment_length*num_segments].reshape(num_segments, segment_length)

    # Split the data into training and testing. Get train/test split parameters from kwargs.
    test_size = kwargs.get("test_size", 0.2)
    shuffle = kwargs.get("shuffle", True)
    random_state = kwargs.get("random_state", None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=random_state)

    # Convert to pytorch tensors
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    # We need to reshape the y data to be 3D. The last dimension is 1 because we only have one output.
    y_train = y_train.unsqueeze(-1)
    y_test = y_test.unsqueeze(-1)

    if x_pipeline is not None:
        X_train = x_pipeline.fit_transform(X_train)
        X_test = x_pipeline.transform(X_test)
    if y_pipeline is not None:
        y_train = y_pipeline.fit_transform(y_train)
        y_test = y_pipeline.transform(y_test)

    # Create dataloaders for the training and testing data
    batch_size = kwargs.get("batch_size", 64)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Return the dataloaders and the pipelines (if any)
    return_vals = [train_loader, test_loader]
    if x_pipeline is not None:
        return_vals.append(x_pipeline)
    if y_pipeline is not None:
        return_vals.append(y_pipeline)

    return tuple(return_vals)


def load_data(iso: str, batch_size: int = 64, segment_length: int = 24, **kwargs):
    # MISO data regressors loaded as [TOTALLOAD, NGPRICE, WIND, SOLAR]
    # We want to scale TOTALLOAD and NGPRICE to be centered around 0 and scaled by the IQR (robust scaling).
    # WIND and SOLAR are already between 0 and 1, so we won't do any additional scaling.

    # We may also be including the effective capacities of the generators in the ISO, which will be
    # a number between 0 and 1, but we don't know how many generator types there are yet. Our hacky
    # way to deal with this is going to be to read the first line of the data file and count the number
    # of commas.
    transformers = [('robust_scaler', RobustScaler(), [0, 1])]
    if kwargs.get("include_capacities", False):
        possible_files = [f"data/{iso.upper()}/{iso.lower()}_generation.csv", f"data/{iso.upper()}/fuelmix.csv"]
        for file in possible_files:
            if not os.path.exists(file):
                continue
            with open(f"data/{iso.upper()}/fuelmix.csv") as f:
                first_line = f.readline()
                n_generators = first_line.count(",")
            break  # if we've found a valid file, use just that one and break
        n_vre = 1 if iso.lower() == "miso" else 2
        cap_cols = list(range(2 + n_vre, 2 + n_vre + n_generators))
        transformers.append(('minmax_scaler', MinMaxScaler(), cap_cols))
    xtrans = InvertibleColumnTransformer(transformers, remainder='passthrough')

    # For the electricity price, we want first center and scale the data using robust scaling, then
    # apply an arcsinh transformation to squash the peaks of the data. Since the arcsinh function is
    # approximately linear around 0, this will effect the data near the median much less than the data
    # in the tails.
    ytrans = make_pipeline(RobustScaler(), FunctionTransformer(np.arcsinh, np.sinh, check_inverse=False))

    # Wrap the transformers in the TensorTransformer class to we can work with batched tensors with
    # less hassle.
    xtrans = BatchedTransformer(xtrans)
    ytrans = BatchedTransformer(ytrans)

    train_loader, test_loader, xtrans, ytrans = get_dataloaders(iso=iso,
                                                                segment_length=segment_length,
                                                                x_pipeline=xtrans,
                                                                y_pipeline=ytrans,
                                                                batch_size=batch_size,
                                                                **kwargs)
    return train_loader, test_loader, xtrans, ytrans
