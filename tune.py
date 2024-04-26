"""
Hyperparameter tuning for deep learning models.
"""
import optuna
import torch

from models import SequenceEncoder, SequenceDecoder, Seq2Seq, MLP
from trainer import SupervisedTrainer
from dataloader import load_data


def define_model(trial: optuna.Trial):
    """
    Defines a time series deep learning model based on LSTM, RNN, or GRU with hyperparameters tuned
    by Optuna.

    :param trial: optuna.Trial: The Optuna trial object
    :return: torch.nn.Module: The deep learning model
    """
    # Constant parameters
    n_input_vars = 3
    n_target_vars = 1
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


def tune_model():
    """"
    Tunes the hyperparameters of a deep learning model using Optuna. At the moment, there are 10
    parameters to tune: 8 model hyperparameters and 2 optimizer hyperparameters.
    """
    train_loader, test_loader, xtrans, ytrans = load_data()

    def objective(trial: optuna.Trial):
        model = define_model(trial)  # 8 tuned model hyperparameters

        # 2 tuned optimizer hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        criterion = torch.nn.MSELoss()

        # device = "cpu" if model.encoder.hidden_size < 512 else "mps"
        device = "cpu"

        trainer = SupervisedTrainer(model, criterion, optimizer, device)
        results = trainer.train(train_loader=train_loader,
                                test_loader=test_loader,
                                n_epochs=200)
        test_rmse = trainer.evaluate(test_loader, ytrans) ** 0.5

        return test_rmse

    storage = "sqlite:///miso.db"
    study = optuna.create_study(storage=storage, study_name="MISO", direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=500)


if __name__ == "__main__":
    tune_model()
