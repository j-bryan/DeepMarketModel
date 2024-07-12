import numpy as np
import pandas as pd
import fire
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from torchtransformers.preprocessing import StandardScaler, FunctionTransformer, RobustScaler, Normalizer
from torchtransformers.pipeline import Pipeline
from torchtransformers.compose import ColumnTransformer, TransformedTargetRegressor
from torchtransformers.regression import Regressor

import optuna
import torch
from tqdm import tqdm
import os
import pickle


class PinballLoss(torch.nn.Module):
    """
    Calculates the quantile loss function.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensor
    """
    def __init__(self, quantiles: torch.Tensor | None = None):
        super(PinballLoss, self).__init__()
        if quantiles is None:
            quantiles = torch.tensor([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
        # Keeps the quantiles tensor on the same device as everything else and will be a part of
        # the state dict, but it's not a parameter that needs to be optimized.
        self.register_buffer("quantiles", quantiles)

    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        error = target - pred
        upper =  self.quantiles * error
        lower = (self.quantiles - 1) * error

        losses = torch.max(lower, upper)
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss


def load_data(iso: str = "ERCOT"):
    # Time series input data
    # time_series = pd.read_csv(f"data/{iso.upper()}/{iso.lower()}_raw.csv", index_col=0)
    # We're intentionally using the scaled data (not the raw data) for this model so that it's
    # compatible with the time series generation data.
    assert iso == "ERCOT", "Only ERCOT data is available for this example"

    # time_series = pd.read_csv(f"data/ERCOT/ercot_eia_rescaled.csv", index_col=0)
    time_series = pd.read_csv(f"data/ERCOT/ercot_eia_rescaled.csv", index_col=0)
    y = time_series.pop("PRICE").to_numpy()
    X = time_series[["TOTALLOAD", "WIND", "SOLAR"]].to_numpy()

    # Context data, saved as monthly values. The time series data is hourly and will be segmented
    # into days, so the context data needs to be repeated for each day in the month.
    # monthly_context = pd.read_csv(f"data/{iso.upper()}/monthly_context_rescaled.csv", index_col=0)
    monthly_context = pd.read_csv(f"data/{iso.upper()}/monthly_context_rescaled.csv", index_col=0)
    # monthly_context.pop("NGPRICE")  # Let's try without NGPRICE for now
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
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mape = torch.mean(torch.abs((y_true - y_pred) / y_true) < perc, dtype=torch.float).item()
    return mape


class FFNN(torch.nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers, n_units, dropout=0.5):
        super(FFNN, self).__init__()
        if n_hidden_layers < 1:
            raise ValueError("FFNN must have at least one hidden layer")

        layers = []
        layers.append(torch.nn.Linear(input_size, n_units))
        layers.append(torch.nn.SiLU())
        layers.append(torch.nn.Dropout(dropout))
        for _ in range(n_hidden_layers - 1):
            layers.append(torch.nn.Linear(n_units, n_units))
            layers.append(torch.nn.SiLU())
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(n_units, output_size))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def main(
    n_trials: int = 128,
    epochs: int = 1000,
    batch_size: int = 128
):
    model_type = "FFNN"

    # Segmenting, splitting, and re-flattening the data should give us the same train/test/validate
    # split that the context_tune.py script uses but formatted into hours for the baseline models.
    X, y, context = load_data("ERCOT")
    context = np.repeat(context, 24, axis=0)
    X = np.hstack([X, context])
    X = segment_array(X, 24)
    y = segment_array(y, 24)

    # Train/test/validation split that is 70/20/10
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size=0.333, shuffle=True, random_state=42)

    # Re-flatten the data
    X_train = X_train.reshape(-1, X_train.shape[2])
    X_test = X_test.reshape(-1, X_test.shape[2])
    X_validate = X_validate.reshape(-1, X_validate.shape[2])
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    y_validate = y_validate.reshape(-1)

    def objective(trial: optuna.trial.Trial, return_model: bool = False):
        n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
        n_units = trial.suggest_categorical("n_units", [32, 64, 128, 256, 512, 1024])

        # If the a model with these parameters is already in the study, return the score
        try:
            if not isinstance(trial, optuna.trial.FixedTrial):
                states_to_consider = (optuna.trial.TrialState.COMPLETE,)
                trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
                # Check whether we already evaluated the sampled `(x, y)`.
                for t in reversed(trials_to_consider):
                    if trial.params == t.params:
                        # Use the existing value as trial duplicated the parameters.
                        print("Duplicate trial found. Returning existing value.")
                        return t.value
        except:
            print("Error checking for existing trial. Continuing...")

        device = "cpu"

        #=============
        #    Model
        #=============

        # Main FFNN model
        ffnn = FFNN(
            input_size=X_train.shape[1],
            output_size=1,
            n_hidden_layers=n_hidden_layers,
            n_units=n_units
        ).to(device)

        # Pipeline for the model and its inputs
        regressor = Pipeline([
            ("scaler", ColumnTransformer([
                    ("load_scaler", StandardScaler(), [0]),
                    ("context_normalizer", Normalizer(norm="l1"), slice(3, X_train.shape[1] - 1)),
                    ("ngprice_scaler", RobustScaler(), [-1])
                ], remainder="passthrough")),
            ("ffnn", Regressor(model=ffnn,
                               optimizer=torch.optim.Adam,
                               optimizer__lr=5e-4,
                               optimizer__weight_decay=0.0,
                               loss_fn=PinballLoss(device=device),
                               device=device,
                               epochs=epochs,
                               batch_size=batch_size))
        ])
        # Pipeline for the output
        output_scaler = Pipeline([
            ("robust", RobustScaler()),
            ("arcsinh", FunctionTransformer(torch.arcsinh, torch.sinh, validate=False))
        ])

        X_train_t = torch.tensor(X_train).float().to(device)
        y_train_t = torch.tensor(y_train).float().unsqueeze(-1).to(device)

        # Model with transformed features and target
        model = TransformedTargetRegressor(regressor, output_scaler).fit(X_train_t, y_train_t)
        plt.plot(model.regressor.get("ffnn").losses)
        plt.show()

        if return_model:
            return model

        y_test_pred = model.predict(torch.tensor(X_test).float().to(device))
        test_score = pinball(y_test_pred, torch.tensor(y_test).float().to(device)).item()

        return test_score

    # storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("baseline_context_norm_retune.log"))
    # storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("baseline.log"))
    # storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("july10_ffnn.log"))
    # storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("july11_ffnn.log"))

    # study = optuna.create_study(direction="minimize", study_name=f"ffnn", storage=storage, load_if_exists=True)
    # study.optimize(objective, n_trials=n_trials)

    dfs = {
        "train":    pd.DataFrame(),
        "test":     pd.DataFrame(),
        "validate": pd.DataFrame()
    }
    dfs["train"]["True"] = y_train.ravel()
    dfs["test"]["True"] = y_test.ravel()
    dfs["validate"]["True"] = y_validate.ravel()

    # best_params = study.best_params
    best_params = {
        "n_hidden_layers": 3,
        "n_units": 1024
    }
    fixed = optuna.trial.FixedTrial(best_params)
    best_model = objective(fixed, return_model=True)

    # Evaluate best model on all data sets
    best_model = best_model.to("cpu")
    X_train = torch.tensor(X_train).float().to("cpu")
    X_test = torch.tensor(X_test).float().to("cpu")
    X_validate = torch.tensor(X_validate).float().to("cpu")
    y_train_pred = best_model.predict(X_train).squeeze()
    y_test_pred  = best_model.predict(X_test).squeeze()
    y_validate_pred = best_model.predict(X_validate).squeeze()

    y_train = torch.tensor(y_train).float().to("cpu").squeeze()
    y_test = torch.tensor(y_test).float().to("cpu").squeeze()
    y_validate = torch.tensor(y_validate).float().to("cpu").squeeze()

    # Huber loss
    train_score = torch.nn.functional.huber_loss(y_train_pred, y_train).item()
    test_score = torch.nn.functional.huber_loss(y_test_pred, y_test).item()
    validate_score = torch.nn.functional.huber_loss(y_validate_pred, y_validate).item()
    print(f"{model_type:<10}{'Huber Loss':<20}{train_score:12.4f}{test_score:12.4f}{validate_score:12.4f}")

    # MAPE < 20%
    train_score = within_perc_mape(y_train, y_train_pred)
    test_score = within_perc_mape(y_test, y_test_pred)
    validate_score = within_perc_mape(y_validate, y_validate_pred)
    print(f"{model_type:<10}{'MAPE < 20%':<20}{train_score:12.4f}{test_score:12.4f}{validate_score:12.4f}")

    # Pinball loss
    pinball = PinballLoss(device="cpu")
    train_score = pinball(y_train_pred, y_train).item()
    test_score = pinball(y_test_pred, y_test).item()
    validate_score = pinball(y_validate_pred, y_validate).item()
    print(f"{model_type:<10}{'Pinball Loss':<20}{train_score:12.4f}{test_score:12.4f}{validate_score:12.4f}")


    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    ax[0].plot(y_train, label="True")
    ax[1].plot(y_test, label="True")
    ax[2].plot(y_validate, label="True")

    ax[0].plot(y_train_pred.detach().cpu().numpy(), label=model_type)
    ax[1].plot(y_test_pred.detach().cpu().numpy(), label=model_type)
    ax[2].plot(y_validate_pred.detach().cpu().numpy(), label=model_type)

    ax[0].set_title("Train")
    ax[1].set_title("Test")
    ax[2].set_title("Validate")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.savefig("july11_samples.png")

    # # Save the model
    os.makedirs("models", exist_ok=True)
    with open(f"models/ffnn.pk", "wb") as f:
        pickle.dump(best_model, f)

    dfs["train"][model_type] = y_train_pred.detach().cpu().numpy().ravel()
    dfs["test"][model_type] = y_test_pred.detach().cpu().numpy().ravel()
    dfs["validate"][model_type] = y_validate_pred.detach().cpu().numpy().ravel()

    for k, df in dfs.items():
        df.to_csv(f"{k}_predictions.csv")


if __name__ == "__main__":
    fire.Fire(main)
