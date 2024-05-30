"""
Hyperparameter tuning for deep learning models.
"""
import json

import optuna
import torch
import fire

from models import SequenceEncoder, SequenceDecoder, ContextHiddenDecoder, ContextInputDecoder, ContextSeq2Seq
from trainer import SupervisedTrainer
from dataloader import load_data, get_num_context_vars


def define_model(trial: optuna.Trial, n_input_vars: int, n_target_vars: int, n_context_vars: int) -> torch.nn.Module:
    """
    Defines a time series deep learning model based on LSTM, RNN, or GRU with hyperparameters tuned
    by Optuna.

    :param trial: optuna.Trial: The Optuna trial object
    :return: torch.nn.Module: The deep learning model
    """
    # Constant parameters
    batch_first = True

    # Tuned hyperparameters parameters
    layer_type = "GRU"

    # Common model parameters
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])

    # Encoder parameters
    # e_dropout = trial.suggest_float("e_dropout", 0.0, 0.3)
    e_dropout = 0.0

    # Decoder parameters
    d_feed_previous = True
    # d_dropout = trial.suggest_float("d_dropout", 0.0, 0.3)
    d_dropout = 0.0

    # Define encoder model
    encoder_context = "none"
    n_encoder_input_vars = n_input_vars + n_context_vars * (encoder_context == "input")
    encoder = SequenceEncoder(layer_type=layer_type,
                            input_size=n_encoder_input_vars,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=e_dropout,
                            bidirectional=bidirectional)

    # Define decoder readout model
    num_directions = 2 if bidirectional else 1

    # Define decoder model
    decoder_context = trial.suggest_categorical("decoder_context", ["hidden", "input", "none"])
    if decoder_context == "hidden":
        output_model = torch.nn.Linear((hidden_size + n_context_vars) * num_directions, n_target_vars)
        decoder = ContextHiddenDecoder(layer_type=layer_type,
                                    input_size=n_target_vars,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    conditional_size=n_context_vars,
                                    batch_first=batch_first,
                                    dropout=d_dropout,
                                    bidirectional=bidirectional,
                                    output_model=output_model,
                                    feed_previous=d_feed_previous)
    elif decoder_context == "input":
        output_model = torch.nn.Linear(hidden_size * num_directions, n_target_vars)
        decoder = ContextInputDecoder(layer_type=layer_type,
                                   input_size=n_target_vars,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   conditional_size=n_context_vars,
                                   batch_first=batch_first,
                                   dropout=d_dropout,
                                   bidirectional=bidirectional,
                                   output_model=output_model,
                                   feed_previous=d_feed_previous)
    else:
        output_model = torch.nn.Linear(hidden_size * num_directions, n_target_vars)
        decoder = SequenceDecoder(layer_type=layer_type,
                                  input_size=n_target_vars,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=batch_first,
                                  dropout=d_dropout,
                                  bidirectional=bidirectional,
                                  output_model=output_model,
                                  feed_previous=d_feed_previous)

    model = ContextSeq2Seq(encoder, decoder, encoder_context != "none", decoder_context != "none", n_context_vars)

    return model


def tune_model(
        iso: str = "ERCOT",
        study_name: str = "%ISO%",  # Default to the ISO name
        n_trials: int = 10,  # Number of optuna trials
        load_if_exists: bool = False,  # Load the study if it exists
        storage: str = "sqlite:///market_parameter_tuning.db",  # Storage URL for the Optuna study
        no_storage: bool = False,  # Do not store the Optuna study
        n_jobs: int = 1,  # Number of threads to use in the Optuna study. Recommend using process-based parallelism instead for CPU-bound tasks.
        batch_size: int = 64,  # Batch size for the data loader
        segment_length: int = 24,  # Segment length for the data loader
        epochs: int = 30,  # Number of epochs to train each model
        device: str = "cuda",  # Device to run the model on
    ):
    """"
    Tunes the hyperparameters of a deep learning model using Optuna. At the moment, there are 10
    parameters to tune: 8 model hyperparameters and 2 optimizer hyperparameters.
    """
    if "%ISO%" in study_name:
        study_name = study_name.replace("%ISO%", iso.upper())

    if no_storage:
        print("No storage specified. Study results will not be saved.")
        storage = None

    # Load data. We extract the dimensionality of the input and target variables from the data loader.
    train_loader, test_loader, xtrans, ytrans = load_data(iso, batch_size, segment_length, include_capacities=True)
    n_conditional_vars = get_num_context_vars(iso)
    train_tensor, target_tensor = next(iter(train_loader))
    n_input_vars = train_tensor.shape[-1] - n_conditional_vars
    n_target_vars = target_tensor.shape[-1]

    # Define the objective function for Optuna. This dynamically defines the model based on parameters
    # suggested by the Optuna trial object. The hyperparameters are optimized using RMSE because RMSE
    # is interpretable (same units as the target variable) and is sensitive to outlier price values,
    # which are important to plant economics.
    def objective(trial: optuna.Trial):
        model = define_model(trial, n_input_vars, n_target_vars, n_conditional_vars)  # 8 tuned model hyperparameters

        # 2 tuned optimizer hyperparameters
        # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        # weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        learning_rate = 3e-3
        weight_decay = 1e-5
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
            # test_rmse = trainer.evaluate(test_loader, ytrans) ** 0.5
            test_rmse = results.best_test_loss ** 0.5
        except ValueError:  # Catch things like NaN and inf problems so the whole study doesn't crash
            test_rmse = 1e6

        results.plot_losses()

        return test_rmse

    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(storage=storage, study_name=study_name, direction="minimize", load_if_exists=load_if_exists, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)


if __name__ == "__main__":
    fire.Fire(tune_model)
