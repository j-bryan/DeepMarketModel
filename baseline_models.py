import numpy as np
import pandas as pd
import fire

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import optuna
import multiprocessing
import os
import pickle
import platform


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


def main(
    model_type: str,
    n_trials: int = 100,
):
    X, y, context = load_data("ERCOT")
    context = np.repeat(context, 24, axis=0)
    X = np.hstack([X, context])

    # Train/test/validation split that is 70/20/10
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size=0.333, shuffle=True, random_state=42)

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
        else:
            raise ValueError(f"Unknown model type {model_type}")

        standard_scaler_cols = list(range(X_train.shape[1]))
        standard_scaler_cols.pop(2)  # Use other scaler for SOLAR
        input_scaler = ColumnTransformer([
            ("load_wind_scaler", StandardScaler(), [0, 1]),
            ("solar_scaler", MinMaxScaler(), [2])
        ], remainder="passthrough")
        output_scaler = Pipeline([
            ("robust", RobustScaler()),
            ("sigmoid", FunctionTransformer(func=np.arcsinh, inverse_func=np.sinh, validate=False))
        ])
        context_scaler = StandardScaler()

        regressor_pipeline = Pipeline([
            ("scaler", input_scaler),
            ("model", model)
        ])
        regressor = TransformedTargetRegressor(regressor_pipeline, output_scaler, context_scaler)

        regressor.fit(X_train, y_train)

        y_test_pred = regressor.predict(X_test)
        test_mape = np.mean(np.abs(y_test - y_test_pred) / y_test)

        return test_mape

    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("optuna.log"))
    study = optuna.create_study(direction="minimize", study_name=f"ercot_{model_type}", storage=storage, load_if_exists=True)

    trials_per_worker = n_trials // multiprocessing.cpu_count()
    ctx = multiprocessing.get_context("fork") if platform.system() == "Linux" else multiprocessing.get_context("spawn")
    with ctx.Pool(multiprocessing.cpu_count()) as pool:
        pool.starmap(study.optimize, [(objective, trials_per_worker)] * multiprocessing.cpu_count())

    print("Best parameters:")
    best_params = study.best_params
    print(best_params)
    fixed = optuna.trial.FixedTrial(best_params)
    best_model = objective(fixed, return_model=True)

    # Evaluate best model on all data sets
    y_train_pred = best_model.predict(X_train)
    y_test_pred  = best_model.predict(X_test)
    y_validate_pred = best_model.predict(X_validate)

    train_mape = np.mean(np.abs(y_train - y_train_pred) / y_train)
    test_mape = np.mean(np.abs(y_test - y_test_pred) / y_test)
    validate_mape = np.mean(np.abs(y_validate - y_validate_pred) / y_validate)

    print("Calculating metrics for best model.")
    print(f"... Train MAPE: {train_mape:.4}")
    print(f"... Test  MAPE: {test_mape:.4}")
    print(f"... Valid MAPE: {validate_mape:.4}")

    # Save the model
    os.makedirs("models", exist_ok=True)
    with open(f"ercot_{model_type.lower()}_train.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Retrain the model on all data
    print("Retraining best model on all data.")
    best_model.fit(X, y)
    y_train_pred_final = best_model.predict(X_train)
    y_test_pred_final  = best_model.predict(X_test)
    y_validate_pred_final = best_model.predict(X_validate)

    train_mape_final = np.mean(np.abs(y_train - y_train_pred_final) / y_train)
    test_mape_final = np.mean(np.abs(y_test - y_test_pred_final) / y_test)
    validate_mape_final = np.mean(np.abs(y_validate - y_validate_pred_final) / y_validate)

    print("Calculating metrics for final model.")
    print(f"... Train MAPE: {train_mape_final:.4}")
    print(f"... Test  MAPE: {test_mape_final:.4}")
    print(f"... Valid MAPE: {validate_mape_final:.4}")

    # Save the model
    os.makedirs("models", exist_ok=True)
    with open(f"ercot_{model_type.lower()}_all.pkl", "wb") as f:
        pickle.dump(best_model, f)


if __name__ == "__main__":
    fire.Fire(main)
