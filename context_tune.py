import numpy as np
import pandas as pd
import fire

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
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
    y = time_series.pop("PRICE").to_numpy()
    X = time_series.to_numpy()

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
    n_trials: int = 100,
    epochs: int = 100,
    batch_size: int = 128
):
    X, y, context = load_data("ERCOT")
    X = segment_array(X, segment_length=24)
    y = segment_array(y, segment_length=24)

    # Train/test/validation split that is 70/20/10
    X_train, X_test, y_train, y_test, context_train, context_test = train_test_split(X, y, context, test_size=0.3, shuffle=True)
    X_test, X_validate, y_test, y_validate, context_test, context_validate = train_test_split(X_test, y_test, context_test, test_size=0.333, shuffle=True)

    if len(X_train) % batch_size != 0:
        pass

    # Save the validation data for later
    np.savez("data/ERCOT/ercot_validation.npz", X=X_validate, y=y_validate, context=context_validate)

    input_state_size = X.shape[-1]
    output_state_size = y.shape[-1]
    context_size = context.shape[-1]

    def objective(trial: optuna.Trial):
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        D = 2 if bidirectional else 1
        hidden_size = trial.suggest_int("hidden_size", 16, 64)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        teacher_forcing = trial.suggest_categorical("teacher_forcing", [True, False])
        layer_type = "GRU"

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
                                criterion=torch.nn.MSELoss(),
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
        context_scaler = StandardScaler()

        regressor_pipeline = ContextPipeline([
            ("scaler", input_scaler),
            ("net", net)
        ])
        regressor = ContextTransformedTargetRegressor(regressor_pipeline, output_scaler, context_scaler)

        regressor.fit(X_train, y_train, context_train, teacher_forcing=teacher_forcing)

        y_test_pred = regressor.predict(X_test, context_test)
        test_mape = np.mean(np.abs(y_test - y_test_pred) / y_test)

        return test_mape

    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("optuna.log"))
    study = optuna.create_study(direction="minimize", study_name="ercot_final", storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    # fixed = optuna.trial.FixedTrial({"bidirectional": False, "hidden_size": 16, "num_layers": 1, "teacher_forcing": False})
    # fixed = optuna.trial.FixedTrial({"bidirectional": True, "hidden_size": 64, "num_layers": 3, "teacher_forcing": False})
    # objective(fixed)


if __name__ == "__main__":
    fire.Fire(main)
