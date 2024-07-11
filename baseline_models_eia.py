import numpy as np
import pandas as pd
import fire

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler, RobustScaler, Normalizer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import optuna
import torch
from skorch.regressor import NeuralNetRegressor
import multiprocessing
import os
import pickle
import platform


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
    def __init__(self, quantiles=torch.tensor([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles.to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """
        error = target.unsqueeze(-1) - pred
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
    context_dates = pd.date_range(end="2022-12-01", periods=monthly_context.shape[0], freq="ME")
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


class FFNN(torch.nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers, n_units):
        super(FFNN, self).__init__()
        if n_hidden_layers < 1:
            raise ValueError("FFNN must have at least one hidden layer")

        layers = []
        layers.append(torch.nn.Linear(input_size, n_units))
        layers.append(torch.nn.ReLU())
        for _ in range(n_hidden_layers - 1):
            layers.append(torch.nn.Linear(n_units, n_units))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(n_units, output_size))
        self.model = torch.nn.Sequential(*layers)

        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                layer.weight = torch.nn.Parameter(layer.weight.double())
                layer.bias = torch.nn.Parameter(layer.bias.double())

    def forward(self, x):
        return self.model(x)


def main(
    model_type: str = "",
    n_trials: int = 128,
    epochs: int = 1000,
    eval: bool = False,
    study_name: str = "ercot"
):
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

    # Tuning objective for a range of baseline models: LR, KNN, RF, GBT
    # These all have different hyperparameters, so we'll need to tune them separately.
    model_params = {
        "LR": {
            "loss": ("categorical", ["squared_error", "huber"]),
            "penalty": ("categorical", ["l1", "l2", "elasticnet", None]),
            "alpha": ("loguniform", [1e-5, 1e-1]),
            "l1_ratio": ("uniform", [0, 1])  # Only used by model if penalty is elasticnet
        },
        "KNN": {
            "n_neighbors": ("int", [1, 20]),
            "weights": ("categorical", ["uniform", "distance"]),
        },
        "RF": {
            "n_estimators": ("int", [10, 250])
        },
        "GBT": {
            "loss": ("categorical", ["squared_error", "huber"]),
            "learning_rate": ("loguniform", [1e-2, 1e0]),
            "n_estimators": ("int", [10, 250])
        },
        "FFNN": {
            "n_hidden_layers": ("int", [1, 3]),
            "n_units": ("categorical", [32, 64, 128, 256, 512])
        }
    }

    def objective(trial: optuna.trial.Trial, return_model: bool = False):
        params = model_params[model_type]

        suggested_params = {}
        for k, v in params.items():
            if v[0] == "categorical":
                suggested_params[k] = trial.suggest_categorical(k, v[1])
            elif v[0] == "int":
                suggested_params[k] = trial.suggest_int(k, v[1][0], v[1][1])
            elif v[0] == "uniform":
                suggested_params[k] = trial.suggest_float(k, v[1][0], v[1][1])
            elif v[0] == "loguniform":
                suggested_params[k] = trial.suggest_float(k, v[1][0], v[1][1], log=True)
            else:
                raise ValueError(f"Unknown parameter type {v[0]}")

        if model_type == "LR":
            model = SGDRegressor(**suggested_params)
        elif model_type == "KNN":
            model = KNeighborsRegressor(**suggested_params)
        elif model_type == "RF":
            model = RandomForestRegressor(**suggested_params)
        elif model_type == "GBT":
            model = GradientBoostingRegressor(**suggested_params)
        elif model_type == "FFNN":
            model = NeuralNetRegressor(
                FFNN,
                module__input_size=X_train.shape[1],
                module__output_size=1,
                module__n_hidden_layers=suggested_params["n_hidden_layers"],
                module__n_units=suggested_params["n_units"],
                max_epochs=epochs,
                optimizer=torch.optim.Adam,
                optimizer__lr=1e-3,
                criterion=PinballLoss,
                batch_size=256,
                train_split=None
            )
        else:
            raise ValueError(f"Unknown model type {model_type}")

        standard_scaler_cols = list(range(X_train.shape[1]))
        standard_scaler_cols.pop(2)  # Use other scaler for SOLAR
        input_scaler = ColumnTransformer([
            ("lws_scaler", StandardScaler(), slice(0, 3)),
            ("context_normalizer", Normalizer(norm="l1"), slice(3, X_train.shape[1] - 1)),
            ("ngprice_scaler", RobustScaler(), [-1])
        ], remainder="passthrough")
        if model_type == "LR":
            output_scaler = Pipeline([
                ("robust", RobustScaler()),
                ("arcsinh", FunctionTransformer(func=np.arcsinh, inverse_func=np.sinh, validate=False))
            ])
        else:
            output_scaler = RobustScaler()

        regressor_pipeline = Pipeline([
            ("scaler", input_scaler),
            ("model", model)
        ])
        regressor = TransformedTargetRegressor(regressor=regressor_pipeline, transformer=output_scaler)

        # if model_type == "FFNN":
        #     regressor.fit(X_train, y_train.reshape(-1, 1))
        # else:
        #     regressor.fit(X_train, y_train)

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

        if model_type != "FFNN":
            regressor.fit(X_train, y_train)
            y_test_pred = regressor.predict(X_test)
        else:
            X_train_t = input_scaler.fit_transform(X_train)
            y_train_t = output_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            X_test_t = input_scaler.transform(X_test)

            regressor.fit(X_train_t, y_train_t.reshape(-1, 1))

            y_test_pred_t = regressor.predict(X_test_t)
            y_test_pred = output_scaler.inverse_transform(y_test_pred_t.reshape(-1, 1)).flatten()

        if return_model:
            return regressor

        test_score = torch.nn.functional.huber_loss(torch.tensor(y_test_pred), torch.tensor(y_test)).item()

        return test_score

    # storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("baseline_context_norm_retune.log"))
    # storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("baseline.log"))
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("july9.log"))

    if not eval:
        study = optuna.create_study(direction="minimize", study_name=f"{study_name}_{model_type}", storage=storage, load_if_exists=True)
        study.optimize(objective, n_trials=n_trials)
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 8))
        ax[0].plot(y_train, label="True")
        ax[1].plot(y_test, label="True")
        ax[2].plot(y_validate, label="True")
        ax[0].set_title("Train")
        ax[1].set_title("Test")
        ax[2].set_title("Validate")

        dfs = {
            "train":    pd.DataFrame(),
            "test":     pd.DataFrame(),
            "validate": pd.DataFrame()
        }
        dfs["train"]["True"] = y_train.ravel()
        dfs["test"]["True"] = y_test.ravel()
        dfs["validate"]["True"] = y_validate.ravel()

        # for model_type in ["LR", "KNN", "RF", "GBT"]:
        for model_type in ["FFNN"]:
            # print(model_type)
            # study = optuna.load_study(study_name=f"{study_name}_{model_type}", storage=storage)

            # best_params = study.best_params
            best_params = {
                "n_hidden_layers": 3,
                "n_units": 512
            }
            fixed = optuna.trial.FixedTrial(best_params)
            best_model = objective(fixed, return_model=True)
            best_model.fit(X_train, y_train)

            # Evaluate best model on all data sets
            y_train_pred = best_model.predict(X_train)
            y_test_pred  = best_model.predict(X_test)
            y_validate_pred = best_model.predict(X_validate)

            dfs["train"][model_type] = y_train_pred
            dfs["test"][model_type] = y_test_pred
            dfs["validate"][model_type] = y_validate_pred

            train_score = torch.nn.functional.huber_loss(torch.tensor(y_train_pred), torch.tensor(y_train)).item()
            test_score = torch.nn.functional.huber_loss(torch.tensor(y_test_pred), torch.tensor(y_test)).item()
            validate_score = torch.nn.functional.huber_loss(torch.tensor(y_validate_pred), torch.tensor(y_validate)).item()

            # print("Huber Loss:")
            # print(f"... Train score: {train_score:.4}")
            # print(f"... Test  score: {test_score:.4}")
            # print(f"... Valid score: {validate_score:.4}")
            # print()

            print(f"{model_type:<10}{'Huber Loss':<20}{train_score:12.4f}{test_score:12.4f}{validate_score:12.4f}")

            train_score = within_perc_mape(y_train, y_train_pred)
            test_score = within_perc_mape(y_test, y_test_pred)
            validate_score = within_perc_mape(y_validate, y_validate_pred)

            print(f"{model_type:<10}{'MAPE < 20%':<20}{train_score:12.4f}{test_score:12.4f}{validate_score:12.4f}")
            # print("Within 20%:")
            # print(f"... Train score: {train_score:.4}")
            # print(f"... Test  score: {test_score:.4}")
            # print(f"... Valid score: {validate_score:.4}")
            # print()

            # Save the model
            os.makedirs("models", exist_ok=True)
            with open(f"models/ercot_{model_type.lower()}_train_nocontext.pkl", "wb") as f:
                pickle.dump(best_model, f)

            ax[0].plot(y_train_pred, label=model_type)
            ax[1].plot(y_test_pred, label=model_type)
            ax[2].plot(y_validate_pred, label=model_type)

            # # Retrain the model on all data
            # print("Retraining best model on all data.")
            # best_model.fit(X.reshape(-1, X.shape[2]), y.reshape(-1))
            # y_train_pred_final = best_model.predict(X_train)
            # y_test_pred_final  = best_model.predict(X_test)
            # y_validate_pred_final = best_model.predict(X_validate)

            # # train_score_final = torch.nn.functional.huber_loss(torch.tensor(y_train_pred_final), torch.tensor(y_train)).item()
            # # test_score_final = torch.nn.functional.huber_loss(torch.tensor(y_test_pred_final), torch.tensor(y_test)).item()
            # # validate_score_final = torch.nn.functional.huber_loss(torch.tensor(y_validate_pred_final), torch.tensor(y_validate)).item()
            # train_score_final = within_perc_mape(y_train, y_train_pred_final)
            # test_score_final = within_perc_mape(y_test, y_test_pred_final)
            # validate_score_final = within_perc_mape(y_validate, y_validate_pred_final)

            # print(f"... Train score: {train_score_final:.4}")
            # print(f"... Test  score: {test_score_final:.4}")
            # print(f"... Valid score: {validate_score_final:.4}")
            # print()

            # # Save the model
            # os.makedirs("models", exist_ok=True)
            # with open(f"models/ercot_{model_type.lower()}_all_nocontext.pkl", "wb") as f:
            #     pickle.dump(best_model, f)

        for k, df in dfs.items():
            df.to_csv(f"{k}_predictions.csv")

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
