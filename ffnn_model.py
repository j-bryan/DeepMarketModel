import numpy as np
import pandas as pd
import fire

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler, Normalizer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

import optuna
import torch
from torch import nn, optim
import tqdm


def load_data(iso: str = "ERCOT"):
    # Time series input data
    # time_series = pd.read_csv(f"data/{iso.upper()}/{iso.lower()}_raw.csv", index_col=0)
    # We're intentionally using the scaled data (not the raw data) for this model so that it's
    # compatible with the time series generation data.
    time_series = pd.read_csv(f"data/{iso.upper()}/{iso.lower()}.csv", index_col=0)
    time_series.pop("NGPRICE")
    y = time_series.pop("PRICE").to_numpy()
    X = time_series[["TOTALLOAD", "WIND", "SOLAR"]].to_numpy()

    # Context data, saved as monthly values. The time series data is hourly and will be segmented
    # into days, so the context data needs to be repeated for each day in the month.
    monthly_context = pd.read_csv(f"data/{iso.upper()}/monthly_context.csv", index_col=0)
    monthly_context.pop("NGPRICE")  # Let's try without NGPRICE for now
    context_dates = pd.date_range(end="2022-12-01", periods=monthly_context.shape[0], freq="M")
    context = []
    for i, date in enumerate(context_dates):
        context.append(np.tile(monthly_context.iloc[i].to_numpy(), (date.days_in_month, 1)))
    context = np.vstack(context)

    return X, y, context


def segment_array(x, segment_length):
    if len(x) % segment_length != 0:
        raise ValueError(f"Data length {len(x)} is not divisible by segment length {segment_length}")
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x.reshape(-1, segment_length, x.shape[1])


def within_perc_mape(y_true, y_pred, perc=0.2):
    mape = np.abs((y_true - y_pred) / y_true)
    return np.mean(mape < perc)


# Define the neural network
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, num_units, num_hidden_layers):
        super(FeedforwardNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, num_units))
        layers.append(nn.LeakyReLU())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(num_units, num_units))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(num_units, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        return self.layers(X)


class NeuralNetRegressor:
    """
    Supposed to work like skorch's NeuralNetRegressor but with a custom training loop. I just
    want to see if I can figure out why the FFNN model isn't working as expected and this gives me
    more control over the training loop.
    """
    def __init__(self, module, optimizer, criterion=nn.HuberLoss(), device="cuda", batch_size=256, epochs=100):
        self.module = module
        self.optimizer = optimizer
        self.criterion = criterion
        self._y_is_1d = False
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "cuda" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.module.to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        if y.ndim == 1:
            self._y_is_1d = True
            y = y.reshape(-1, 1)
        dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        bar = tqdm.trange(self.epochs)
        for epoch in bar:
            total_loss = 0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                y_pred = self.module(X_batch.to(self.device))
                loss = torch.nn.functional.huber_loss(y_pred, y_batch.to(self.device))
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            bar.set_postfix(loss=total_loss / len(loader))

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        yt = self.module(X).detach().cpu().numpy()
        if self._y_is_1d:
            yt = yt.flatten()
        return yt


def main(
    epochs: int = 500,
    batch_size: int = 1024,
    device: str = "cuda"
):
    # Segmenting, splitting, and re-flattening the data should give us the same train/test/validate
    # split that the context_tune.py script uses but formatted into hours for the baseline models.
    X, y, context = load_data("ERCOT")
    context = np.repeat(context, 24, axis=0).astype(np.float32)
    X = np.hstack([X.astype(np.float32), context])
    y = y.astype(np.float32).reshape(-1, 1)
    X = segment_array(X, 24)
    y = segment_array(y, 24).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.333, random_state=42)

    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_test = X_test.reshape(-1, X_test.shape[-1])
    X_val = X_val.reshape(-1, X_val.shape[-1])
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    y_val = y_val.reshape(-1)

    def objective(trial: optuna.trial.Trial):
        # Transformations for input data
        preprocessor = ColumnTransformer(
            transformers=[
                ('scale_col_0', StandardScaler(), [0]),
                ('pass_cols_1_2', 'passthrough', [1, 2]),
                ('l1_norm_rest', Normalizer(norm="l1"), slice(3, None))
            ]
        )

        # Transformations for output data
        output_transformer = Pipeline([
            ('robust_scaler', RobustScaler()),
            ('arcsinh', FunctionTransformer(func=np.arcsinh, inverse_func=np.sinh))
        ])

        n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
        n_units = trial.suggest_categorical("n_units", [512, 1024, 2048, 4096])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # Protect against running the same trial multiple times. If it's already been run, return
        # the value from the previous run.
        if not isinstance(trial, optuna.trial.FixedTrial):
            states_to_consider = (optuna.trial.TrialState.COMPLETE,)
            trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
            # Check whether we already evaluated the sampled `(x, y)`.
            for t in reversed(trials_to_consider):
                if trial.params == t.params:
                    # Use the existing value as trial duplicated the parameters.
                    return t.value

        ffnn = FeedforwardNN(X_train.shape[-1], num_hidden_layers=n_hidden_layers, num_units=n_units)
        optimizer = optim.AdamW(ffnn.parameters(), lr=1e-4, weight_decay=weight_decay)
        net = NeuralNetRegressor(ffnn, optimizer, device=device, epochs=epochs, batch_size=batch_size)

        regressor = TransformedTargetRegressor(
            regressor=Pipeline([
                ("scaler", preprocessor),
                ("net", net)
            ]),
            transformer=output_transformer
        )

        # Train the model
        regressor.fit(X_train, y_train.flatten())

        # Example prediction
        y_train_pred = regressor.predict(X_train)
        y_test_pred = regressor.predict(X_test)
        y_val_pred = regressor.predict(X_val)

        # Calculate Huber loss on untransformed data
        train_loss = torch.nn.functional.huber_loss(torch.tensor(y_train_pred), torch.tensor(y_train)).item()
        test_loss = torch.nn.functional.huber_loss(torch.tensor(y_test_pred), torch.tensor(y_test)).item()
        val_loss = torch.nn.functional.huber_loss(torch.tensor(y_val_pred), torch.tensor(y_val)).item()
        print("Huber Loss")
        print(f"... Train: {train_loss:.4f}")
        print(f"... Test: {test_loss:.4f}")
        print(f"... Validation: {val_loss:.4f}")

        # Calculate Within 20% MAPE
        train_mape = within_perc_mape(y_train, y_train_pred, perc=0.2)
        test_mape = within_perc_mape(y_test, y_test_pred, perc=0.2)
        val_mape = within_perc_mape(y_val, y_val_pred, perc=0.2)
        print("Within 20% MAPE")
        print(f"... Train: {train_mape:.4f}")
        print(f"... Test: {test_mape:.4f}")
        print(f"... Validation: {val_mape:.4f}")
        return test_loss

    storage = optuna.storages.JournalStorage(optuna.storage.JournalFileStorage("ffnn.log"))
    study = optuna.create_study(study_name="ffnn", storage=storage, direction="minimize")
    study.optimize(objective, n_trials=128)

    best_params = study.best_params
    print("\n\n\n")
    print("Best Parameters")
    print(best_params)
    fixed_trial = optuna.trial.FixedTrial(best_params)
    objective(fixed_trial)


if __name__ == "__main__":
    fire.Fire(main)
