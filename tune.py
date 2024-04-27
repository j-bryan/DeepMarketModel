"""
Hyperparameter tuning for deep learning models.
"""
import json

import optuna
import torch

from models import SequenceEncoder, SequenceDecoder, Seq2Seq, MLP
from trainer import SupervisedTrainer
from dataloader import load_data


def define_model(trial: optuna.Trial, n_input_vars: int, n_target_vars: int) -> torch.nn.Module:
    """
    Defines a time series deep learning model based on LSTM, RNN, or GRU with hyperparameters tuned
    by Optuna.

    :param trial: optuna.Trial: The Optuna trial object
    :return: torch.nn.Module: The deep learning model
    """
    # Constant parameters
    batch_first = True

    # Tuned hyperparameters parameters
    layer_type = trial.suggest_categorical("recurrent_layer", ["LSTM", "RNN", "GRU"])

    # Common model parameters
    hidden_sizes = [2 ** i for i in range(3, 9)]
    hidden_size = trial.suggest_categorical("hidden_size", hidden_sizes)
    num_layers = trial.suggest_int("e_num_layers", 1, 3)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    teacher_forcing = trial.suggest_categorical("teacher_forcing", [True, False])

    # Encoder parameters
    if num_layers > 1:
        e_dropout = trial.suggest_float("dropout", 0.0, 0.3)
    else:
        e_dropout = 0.0

    # Decoder parameters
    if teacher_forcing:
        d_feed_previous = True
    else:
        d_feed_previous = trial.suggest_categorical("d_feed_previous", [True, False])

    if num_layers > 1:
        d_dropout = trial.suggest_float("d_dropout", 0.0, 0.3)
    else:
        d_dropout = 0.0

    # Define encoder model
    encoder = SequenceEncoder(layer_type=layer_type,
                              input_size=n_input_vars,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              dropout=e_dropout,
                              bidirectional=bidirectional)

    # Define decoder readout model
    num_directions = 2 if bidirectional else 1
    output_model = torch.nn.Linear(hidden_size * num_directions, n_target_vars)

    # Define decoder model
    decoder = SequenceDecoder(layer_type=layer_type,
                              input_size=n_target_vars,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              dropout=d_dropout,
                              bidirectional=bidirectional,
                              output_model=output_model,
                              feed_previous=d_feed_previous)

    model = Seq2Seq(encoder, decoder)
    return model


def tune_model(**kwargs):
    """"
    Tunes the hyperparameters of a deep learning model using Optuna. At the moment, there are 10
    parameters to tune: 8 model hyperparameters and 2 optimizer hyperparameters.
    """
    # Parse kwargs
    iso = kwargs["iso"]
    device = kwargs["device"]
    epochs = kwargs["epochs"]
    study_name = kwargs["study_name"]
    load_if_exists = kwargs["load_if_exists"]
    n_trials = kwargs["n_trials"]
    batch_size = kwargs["batch_size"]
    segment_length = kwargs["segment_length"]
    storage = kwargs["storage"]
    no_storage = kwargs["no_storage"]
    n_jobs = kwargs["n_jobs"]

    if "%ISO%" in study_name:
        study_name = study_name.replace("%ISO%", iso.upper())

    if storage in kwargs and no_storage:
        raise ValueError("Cannot provide both storage and no_storage arguments.")
    if no_storage:
        storage = None

    # Load data. We extract the dimensionality of the input and target variables from the data loader.
    train_loader, test_loader, xtrans, ytrans = load_data(iso, batch_size, segment_length)
    n_input_vars = next(iter(train_loader))[0].shape[-1]
    n_target_vars = next(iter(train_loader))[1].shape[-1]

    # Define the objective function for Optuna. This dynamically defines the model based on parameters
    # suggested by the Optuna trial object. The hyperparameters are optimized using RMSE because RMSE
    # is interpretable (same units as the target variable) and is sensitive to outlier price values,
    # which are important to plant economics.
    def objective(trial: optuna.Trial):
        model = define_model(trial, n_input_vars, n_target_vars)  # 8 tuned model hyperparameters

        # 2 tuned optimizer hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        criterion = torch.nn.MSELoss()

        trainer = SupervisedTrainer(model, criterion, optimizer, device)
        try:
            # Intermediate test results are reported to Optuna. trainer.train() may raise an optuna.TrialPruned
            # exception if the trial is pruned. This is caught by the Optuna study object and dealt
            # with appropriately.
            results = trainer.train(train_loader=train_loader,
                                    test_loader=test_loader,
                                    n_epochs=epochs,
                                    optuna_trial=trial)
            test_rmse = trainer.evaluate(test_loader, ytrans) ** 0.5
        except ValueError:  # Catch things like NaN and inf problems so the whole study doesn't crash
            test_rmse = 1e6

        return test_rmse

    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(storage=storage, study_name=study_name, direction="minimize", load_if_exists=load_if_exists, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tune hyperparameters for deep learning models.")

    #========================
    #   REQUIRED ARGUMENTS
    #========================
    parser.add_argument("--iso", type=str, required=True, help="The ISO to tune model")  # required arguments

    #========================
    #   OPTIONAL ARGUMENTS
    #========================
    # Optuna arguments
    parser.add_argument("--study-name", type=str, default="%ISO%", help="The name of the Optuna study. If not provided, the ISO name will be used.")
    parser.add_argument("--n-trials", type=int, default=10, help="The number of trials to run.")
    parser.add_argument("--load-if-exists", action="store_true", help="Load the study if it exists.")
    parser.add_argument("--storage", type=str, default="sqlite:///market_parameter_tuning.db", help="The storage URL for the Optuna study.")
    parser.add_argument("--no-storage", action="store_true", default=False, help="Do not store the Optuna study.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of threads to use in the Optuna study. "
                        "WARNING: This may be slow due to Python's GIL lock. Consider using process-based parallelism instead!")

    # Data arguments
    parser.add_argument("--batch-size", type=int, default=64, help="The batch size for the data loader.")
    parser.add_argument("--segment-length", type=int, default=24, help="The segment length for the data loader.")

    # Model training arguments
    parser.add_argument("--epochs", type=int, default=30, help="The number of epochs to train each model.")
    parser.add_argument("--device", type=str, default="cpu", help="The device to run the model on.")

    args = vars(parser.parse_args())

    tune_model(**args)
