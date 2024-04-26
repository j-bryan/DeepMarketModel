import os
import datetime
import json
import argparse
import platform
import numpy as np
import pandas as pd
import plotly.express as px

import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from metrics import *

from gru import EncoderGRU, DecoderGRU, EncoderDecoderGRU
from dataloader import get_dataloaders


def train(**kwargs):
    # Hyperparameters
    input_size = kwargs.get('input_size', 3)
    hidden_dim = kwargs.get('hidden_dim', 128)
    n_rnn_layers = kwargs.get('n_rnn_layers', 1)
    dropout = kwargs.get('dropout', 0.0)
    learning_rate = kwargs.get('learning_rate', 0.01)
    n_epochs = kwargs.get('n_epochs', 200)
    batch_size = kwargs.get('batch_size', 64)
    segment_length = kwargs.get('segment_length', 24)
    decay = kwargs.get('decay', 1e-4)

    # Other parameters
    device = kwargs.get('device', None)
    random_seed = kwargs.get('random_seed', 12345)
    model_path = kwargs.get('model_path', None)
    eval_only = kwargs.get('eval_only', False)

    # Load the data. Our utility functions will handle the scaling and batching for us.
    train_loader, test_loader, xtrans, ytrans = load_data(batch_size, segment_length, random_state=random_seed)

    # Initialise each of the models
    model_encoder = EncoderGRU(input_size=input_size,
                                hidden_dim=hidden_dim,
                                batch_size=batch_size,
                                n_layers=n_rnn_layers,
                                bidirectional=False,
                                dropout_p=dropout)

    model_decoder = DecoderGRU(hidden_dim=hidden_dim,
                                output_size=1,
                                batch_size=batch_size,
                                n_layers=n_rnn_layers,
                                forecasting_horizon=segment_length,
                                bidirectional=False,
                                dropout_p=0)

    model_encoder_decoder = EncoderDecoderGRU(encoder_model=model_encoder,
                                                decoder_model=model_decoder,
                                                n_epochs=n_epochs,
                                                lr=learning_rate,
                                                decay=decay,
                                                device=device)
    if model_path is not None:
        model_encoder_decoder.load_state_dict(torch.load(model_path))

    if not eval_only:
        train_losses, test_losses = model_encoder_decoder.trainer(train_loader, test_loader)

    # Set up directory to save model and loss plot
    if not model_path:
        dirname = 'saved_models/MISO_GRU'
    else:
        dirname = os.path.dirname(model_path)

    if os.path.exists(dirname) and dirname == "saved_models/MISO_GRU":  # We're using the default directory
        # If the default directory already exists, we'll just append a number to the end of it
        # See what the highest number in the directory is
        num_existing_dirs = sum([1 for d in os.listdir('saved_models') if 'MISO_GRU' in d])
        dirname += f"_{num_existing_dirs}"

    # If we've trained the model, save the model, its hyperparameters, and some metadata
    if n_epochs > 0 and not eval_only:
        os.makedirs(dirname)

        # Save model state dict
        torch.save(model_encoder_decoder.state_dict(), os.path.join(dirname, 'model.pt'))

        # Save the hyperparameters as a JSON file
        hyperparameters = {
            "input_size": input_size,
            "hidden_dim": hidden_dim,
            "n_rnn_layers": n_rnn_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "forecast_horizon": segment_length,
            "decay": decay,
        }
        with open(os.path.join(dirname, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparameters, f)

        # Write some metadata about the training process to a JSON file. This is useful for tracking
        # how and when the model was trained, outside of the model itself and the hyperparameters.
        metadata = {
            "train_losses": train_losses.tolist(),
            "test_losses": test_losses.tolist(),
            "training_time": model_encoder_decoder.training_time,
            "device": str(model_encoder_decoder.device),
            "device_info": str(platform.uname()),  # contains information about the hardware (OS, system name, etc.)
            "pytorch_version": torch.__version__,
            "training_end_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "random_seed": random_seed
        }
        with open(os.path.join(dirname, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        # Plot the training and testing losses
        losses_path = os.path.join(dirname, 'losses.png')
        plot_losses(train_losses, test_losses, losses_path, loss_name="MSE")

    save_dir = kwargs.get('target_dir', None) or dirname
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate metrics for the training and testing sets and save them to a JSON file
    metrics_path = os.path.join(save_dir, 'metrics.json')
    calculate_metrics(model_encoder_decoder, train_loader, test_loader, ytrans, metrics_path)

    # Plot full time series predictions
    time_series_preds_path = os.path.join(save_dir, 'predictions_time_series.png')
    plot_full_time_series(model_encoder_decoder, xtrans, ytrans, segment_length, time_series_preds_path)

    # Plot training and testing set predictions
    plot_train_test_predictions(model_encoder_decoder, train_loader, test_loader, ytrans, save_dir)


def load_data(batch_size: int = 64, segment_length: int = 24, **kwargs):
    # MISO data regressors loaded as [TOTALLOAD, NGPRICE, WIND, SOLAR]
    # We want to scale TOTALLOAD and NGPRICE to be centered around 0 and scaled by the IQR (robust scaling).
    # WIND and SOLAR are already between 0 and 1, so we won't do any additional scaling.
    xtrans = ColumnTransformer([('robust_scaler', RobustScaler(), [0, 1])], remainder='passthrough')
    # For the electricity price, we want first center and scale the data using robust scaling, then
    # apply an arcsinh transformation to squash the peaks of the data. Since the arcsinh function is
    # approximately linear around 0, this will effect the data near the median much less than the data
    # in the tails.
    ytrans = make_pipeline(RobustScaler(), FunctionTransformer(np.arcsinh, np.sinh))
    train_loader, test_loader, xtrans, ytrans = get_dataloaders(iso='MISO',
                                                                segment_length=segment_length,
                                                                x_pipeline=xtrans,
                                                                y_pipeline=ytrans,
                                                                batch_size=batch_size,
                                                                **kwargs)
    return train_loader, test_loader, xtrans, ytrans


def plot_full_time_series(model, xtrans, ytrans, segment_length, save_path):
    # Plot predictions for the original time series data.
    # We need to load the original data again (it was shuffled by the data loader), and then predict
    # on that data.
    data = pd.read_csv("data/MISO/miso.csv", index_col=0)
    X = data[["TOTALLOAD", "NGPRICE", "WIND"]].to_numpy()
    y_true = data["PRICE"].to_numpy()

    Xtrans = xtrans.transform(X)
    n_segments = len(X) // segment_length
    Xtrans = Xtrans[:segment_length * n_segments].reshape(n_segments, segment_length, -1)
    Xtrans = torch.tensor(Xtrans).float()

    y_pred = model.forward(Xtrans)
    y_pred = y_pred.squeeze().detach().numpy().ravel()
    y_pred = ytrans.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    df_timeseries = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    df_timeseries.index = pd.date_range(start='2018-01-01', periods=len(df_timeseries), freq='H')
    fig = px.line(df_timeseries)
    fig.update_layout(xaxis_title="", yaxis_title="Price ($/MWh)")
    fig.write_image(save_path)


def plot_train_test_predictions(model, train_loader, test_loader, ytrans, dirname):
    # Plot the predictions of the full training set
    train_predictions = model.forward(train_loader.dataset.tensors[0]).detach().numpy()
    train_predictions = np.vstack(train_predictions)
    train_predictions = ytrans.inverse_transform(train_predictions)
    train_actuals = np.vstack(train_loader.dataset.tensors[1].detach().numpy())
    train_actuals = ytrans.inverse_transform(train_actuals)
    train_df = pd.DataFrame({'Actual': train_actuals.ravel(), 'Predicted': train_predictions.ravel()})
    fig = px.line(train_df, title="Training Set Predictions")
    fig.update_layout(xaxis_title="Time", yaxis_title="Price")
    fig.write_image(os.path.join(dirname, 'train_predictions.png'))

    # Plot the predictions of the full testing set
    test_predictions = model.forward(test_loader.dataset.tensors[0]).detach().numpy()
    test_predictions = np.vstack(test_predictions)
    test_predictions = ytrans.inverse_transform(test_predictions)
    test_actuals = np.vstack(test_loader.dataset.tensors[1].detach().numpy())
    test_actuals = ytrans.inverse_transform(test_actuals)
    test_df = pd.DataFrame({'Actual': test_actuals.ravel(), 'Predicted': test_predictions.ravel()})
    fig = px.line(test_df, title="Testing Set Predictions")
    fig.update_layout(xaxis_title="Time", yaxis_title="Price")
    fig.write_image(os.path.join(dirname, 'test_predictions.png'))


def plot_losses(train_losses, test_losses, save_path, loss_name=None):
    # Plot the training and testing losses
    epochs = np.arange(1, len(train_losses) + 1)
    losses_df = pd.DataFrame({'Epoch': epochs, 'Train Loss': train_losses, 'Test Loss': test_losses})
    losses_df.set_index('Epoch', inplace=True)
    fig = px.line(losses_df, title="Train and Test Losses")
    fig.update_layout(xaxis_title="Epoch")
    if loss_name is not None:
        fig.update_layout(yaxis_title=loss_name)
    fig.write_image(save_path)


def calculate_metrics(model, train_loader, test_loader, ytrans, save_path):
    # Run model on full train and test sets to get the predicted values for each
    train_predictions = model.forward(train_loader.dataset.tensors[0]).detach().numpy()
    train_predictions = np.vstack(train_predictions)
    train_predictions = ytrans.inverse_transform(train_predictions)
    train_actuals = np.vstack(train_loader.dataset.tensors[1].detach().numpy())
    train_actuals = ytrans.inverse_transform(train_actuals)
    test_predictions = model.forward(test_loader.dataset.tensors[0]).detach().numpy()
    test_predictions = np.vstack(test_predictions)
    test_predictions = ytrans.inverse_transform(test_predictions)
    test_actuals = np.vstack(test_loader.dataset.tensors[1].detach().numpy())
    test_actuals = ytrans.inverse_transform(test_actuals)

    # Calculate metrics for the training and testing sets
    metrics = {}
    metric_functions = {
        'MSE': mean_squared_error,
        'MAE': mean_absolute_error,
        'MAPE': mean_absolute_percentage_error,
        'Within1%': within_1percent,
        'Within5%': within_5percent,
        'Within10%': within_10percent,
        'Within20%': within_20percent,
        'Within50%': within_50percent
    }
    for metric_name, metric_func in metric_functions.items():
        train_metric = metric_func(train_actuals, train_predictions)
        test_metric = metric_func(test_actuals, test_predictions)
        metrics[f"train_{metric_name}"] = str(train_metric)
        metrics[f"test_{metric_name}"] = str(test_metric)

    with open(save_path, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GRU model on the MISO dataset.")
    parser.add_argument("--device", type=str, default=None, help="The device to train on (cuda, mps, cpu)")
    parser.add_argument("--n-epochs", type=int, default=200, help="The number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=64, help="The batch size to use")
    parser.add_argument("--segment-length", type=int, default=24, help="The number of timesteps to forecast")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="The learning rate to use")
    parser.add_argument("--hidden-dim", type=int, default=128, help="The hidden dimension of the GRU")
    parser.add_argument("--n-rnn-layers", type=int, default=1, help="The number of layers in the GRU")
    parser.add_argument("--dropout", type=float, default=0.0, help="The dropout probability")
    parser.add_argument("--decay", type=float, default=1e-4, help="The weight decay")
    parser.add_argument("--model-path", type=str, default=None, help="The path to a pre-trained model")
    parser.add_argument("--eval-only", action="store_true", help="Only generate plots and metrics, don't train.")
    parser.add_argument("--target-dir", type=str, default=None, help="The directory to save outputs to")
    parser.add_argument("--random-seed", type=int, default=12345, help="The random seed to use")
    args = parser.parse_args()

    train(**vars(args))
