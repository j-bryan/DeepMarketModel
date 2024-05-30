"""
Hyperparameter tuning for deep learning models.
"""
import torch
from models import SequenceEncoder, ContextHiddenDecoder, ContextInputDecoder, ContextSeq2Seq, SequenceDecoder
from trainer import SupervisedTrainer
from dataloader import load_data, get_num_context_vars
from main import calculate_metrics


def define_model(n_input_vars: int, n_target_vars: int, n_context_vars: int, encoder_context: str, decoder_context: str) -> torch.nn.Module:
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
    hidden_size = 64
    num_layers = 3
    bidirectional = True

    # Encoder parameters
    e_dropout = 0.1324

    # Decoder parameters
    d_feed_previous = True
    d_dropout = 4.729e-4

    # Define encoder model
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


def main(**kwargs):
    """"
    Tunes the hyperparameters of a deep learning model using Optuna. At the moment, there are 10
    parameters to tune: 8 model hyperparameters and 2 optimizer hyperparameters.
    """
    # Parse kwargs
    iso = kwargs["iso"]
    device = kwargs["device"]
    epochs = kwargs["epochs"]
    batch_size = kwargs["batch_size"]
    segment_length = kwargs["segment_length"]

    encoder_context = kwargs["encoder"]
    decoder_context = kwargs["decoder"]
    include_capacities = encoder_context != "none" or decoder_context != "none"  # If both are none, we don't need the context variables

    # Load data. We extract the dimensionality of the input and target variables from the data loader.
    train_loader, test_loader, xtrans, ytrans = load_data(iso, batch_size, segment_length, include_capacities=include_capacities, shuffle=False)
    n_conditional_vars = 0 if not include_capacities else get_num_context_vars(iso)
    n_input_vars = next(iter(train_loader))[0].shape[-1] - n_conditional_vars
    n_target_vars = next(iter(train_loader))[1].shape[-1]

    # Define the objective function for Optuna. This dynamically defines the model based on parameters
    # suggested by the Optuna trial object. The hyperparameters are optimized using RMSE because RMSE
    # is interpretable (same units as the target variable) and is sensitive to outlier price values,
    # which are important to plant economics.
    model = define_model(n_input_vars, n_target_vars, n_conditional_vars, encoder_context, decoder_context)  # 8 tuned model hyperparameters

    # 2 tuned optimizer hyperparameters
    learning_rate = 3.729e-3
    weight_decay = 1.11e-5
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
                                verbose=True)
        # test_rmse = trainer.evaluate(test_loader, ytrans) ** 0.5
        test_rmse = results.best_test_loss ** 0.5
    except ValueError:  # Catch things like NaN and inf problems so the whole study doesn't crash
        test_rmse = 1e6

    metrics_name = f"metrics/{iso.upper()}OptNoShuffle_enc{encoder_context.title()}_dec{decoder_context.title()}.json"
    calculate_metrics(model, train_loader, test_loader, ytrans, metrics_name)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Tune hyperparameters for deep learning models.")

    # #========================
    # #   REQUIRED ARGUMENTS
    # #========================
    # parser.add_argument("--iso", type=str, required=True, help="The ISO to tune model")  # required arguments

    # #========================
    # #   OPTIONAL ARGUMENTS
    # #========================
    # # Data arguments
    # parser.add_argument("--batch-size", type=int, default=64, help="The batch size for the data loader.")
    # parser.add_argument("--segment-length", type=int, default=24, help="The segment length for the data loader.")
    # parser.add_argument("--encoder", type=str, default="none", help="The kind of context encoder to use. Options are 'none' and 'input'.")
    # parser.add_argument("--decoder", type=str, default="none", help="The kind of context decoder to use. Options are 'none', 'input', and 'hidden'.")

    # # Model training arguments
    # parser.add_argument("--epochs", type=int, default=30, help="The number of epochs to train each model.")
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="The device to run the model on.")

    # args = vars(parser.parse_args())

    # main(**args)

    args = {
        "iso": "ERCOT",
        "batch_size": 1826,
        "segment_length": 24,
        "device": "cuda",
        "encoder": "none",
        "decoder": "none",
        "epochs": 200,
    }
    encoder_types = ["none", "input"]
    decoder_types = ["none", "input", "hidden"]

    import itertools

    for encoder, decoder in itertools.product(encoder_types, decoder_types):
        args["encoder"] = encoder
        args["decoder"] = decoder
        print(f"Encoder Type: {encoder:<10}   Decoder Type: {decoder:<10}")
        try:
            main(**args)
        except Exception as e:
            print(e)
