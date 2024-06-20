import numpy as np
import pandas as pd
import fire

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer, Normalizer
from sklearn.pipeline import Pipeline
from batched_preprocessing import BatchStandardScaler, BatchMinMaxScaler, BatchColumnTransformer, BatchRobustScaler
from context_wrappers import ContextPipeline, ContextTransformedTargetRegressor, NeuralNetRegressor

from context_models import SequenceEncoder, ContextHiddenDecoder, Seq2Seq

import optuna


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


def main(
    n_trials: int = 100,
    epochs: int = 300,
    batch_size: int = 128
):
    X, y, context = load_data("ERCOT")
    X = segment_array(X, segment_length=24)
    y = segment_array(y, segment_length=24)

    # Train/test/validation split that is 70/20/10
    np.random.seed(42)
    torch.manual_seed(42)
    X_train, X_test, y_train, y_test, context_train, context_test = train_test_split(X, y, context, test_size=0.3, shuffle=True, random_state=42)
    X_test, X_validate, y_test, y_validate, context_test, context_validate = train_test_split(X_test, y_test, context_test, test_size=0.333, shuffle=True, random_state=42)

    if batch_size > X_train.shape[0]:
        batch_size = X_train.shape[0]

    input_state_size = X.shape[-1]
    output_state_size = y.shape[-1]
    context_size = context.shape[-1]

    def objective(trial: optuna.Trial, return_model: bool = False):
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        D = 2 if bidirectional else 1
        hidden_size = trial.suggest_int("hidden_size", 16, 64)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        teacher_forcing = trial.suggest_categorical("teacher_forcing", [True, False])
        layer_type = trial.suggest_categorical("layer_type", ["GRU", "LSTM"])

        encoder_params = {
            "layer_type": layer_type,
            "input_size": input_state_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": bidirectional
        }
        decoder_params = {
            "layer_type": layer_type,
            "input_size": output_state_size,
            "hidden_size": hidden_size + context_size,
            "num_layers": num_layers,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": bidirectional,
            "output_model": torch.nn.Linear((hidden_size + context_size) * D, output_state_size),
            "feed_previous": True
        }

        encoder = SequenceEncoder(**encoder_params)
        decoder = ContextHiddenDecoder(**decoder_params)
        model = Seq2Seq(encoder, decoder)

        optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        net = NeuralNetRegressor(model=model,
                                optimizer=optim,
                                criterion=torch.nn.HuberLoss(),
                                epochs=epochs,
                                batch_size=batch_size)

        input_scaler = BatchColumnTransformer([
            ("load_wind_scaler", BatchStandardScaler(), [0, 1]),
            ("solar_scaler", BatchMinMaxScaler(), [2])
        ], remainder="passthrough")
        output_scaler = Pipeline([
            ("robust", BatchRobustScaler()),
            ("sigmoid", FunctionTransformer(func=np.arcsinh, inverse_func=np.sinh, validate=False))
        ])
        context_scaler = Normalizer()

        regressor_pipeline = ContextPipeline([
            ("scaler", input_scaler),
            ("net", net)
        ])
        regressor = ContextTransformedTargetRegressor(regressor_pipeline, output_scaler, context_scaler)

        regressor.fit(X_train, y_train, context_train, teacher_forcing=teacher_forcing)

        if return_model:
            return regressor

        y_test_pred = regressor.predict(X_test, context_test)
        test_score = torch.nn.functional.huber_loss(torch.tensor(y_test_pred), torch.tensor(y_test)).item()

        return test_score

    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("seq2seq.log"))
    study = optuna.create_study(direction="minimize", study_name="ercot", storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)

    print("Best parameters:")
    best_params = study.best_params
    print(best_params)
    fixed = optuna.trial.FixedTrial(best_params)
    best_model = objective(fixed, return_model=True)

    # Evaluate best model on all data sets
    y_train_pred = best_model.predict(X_train, context_train)
    y_test_pred  = best_model.predict(X_test, context_test)
    y_validate_pred = best_model.predict(X_validate, context_validate)

    train_score = torch.nn.functional.huber_loss(torch.tensor(y_train_pred), torch.tensor(y_train)).item()
    test_score = torch.nn.functional.huber_loss(torch.tensor(y_test_pred), torch.tensor(y_test)).item()
    validate_score = torch.nn.functional.huber_loss(torch.tensor(y_validate_pred), torch.tensor(y_validate)).item()

    print("Huber Loss:")
    print(f"... Train MAPE: {train_score:.4}")
    print(f"... Test  MAPE: {test_score:.4}")
    print(f"... Valid MAPE: {validate_score:.4}")

    train_score = within_perc_mape(y_train, y_train_pred)
    test_score = within_perc_mape(y_test, y_test_pred)
    validate_score = within_perc_mape(y_validate, y_validate_pred)

    print("Within 20% MAPE:")
    print(f"... Train: {train_score:.4}")
    print(f"... Test: {test_score:.4}")
    print(f"... Valid: {validate_score:.4}")


if __name__ == "__main__":
    fire.Fire(main)
