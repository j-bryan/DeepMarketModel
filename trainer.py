import numpy as np
import torch
from tqdm import tqdm, trange


class SupervisedTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 criterion,  # torch loss function
                 optimizer: torch.optim.Optimizer,
                 device: str | torch.device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # Move the model to the device
        self.model.to(self.device)

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              n_epochs: int,
              teacher_forcing: bool = False,
              teacher_forcing_ratio: float = 0.1,
              test_loader: torch.utils.data.DataLoader | None = None,
              verbose: bool = False,
              save_best: bool = False,
              save_path: str | None = None,
              save_every: int | None = None,
              optuna_trial=None):
        """
        Train the model on the training data. If test_loader is provided, evaluate the model on the
        test data after each epoch.

        :param train_loader: torch.utils.data.DataLoader: The training data loader
        :param n_epochs: int: The number of epochs to train for
        :param teacher_forcing: bool: Whether to use teacher forcing during training
        :param test_loader: torch.utils.data.DataLoader: The test data loader
        :param verbose: bool: Whether to print training information
        :param save_best: bool: Whether to save the best model
        :param save_path: str: The path to save the model
        :param save_every: int: Save the model every n epochs
        :param optuna_trial: optuna.Trial: The Optuna trial object. Used for reporting intermediate
                             results to Optuna. Must also provide test_loader to use this feature.
        """
        if optuna_trial is not None and test_loader is None:
            raise ValueError("Must provide test_loader if using Optuna trial.")
        elif optuna_trial is not None:
            import optuna  # lazy import because not everyone will be using Optuna

        self.teacher_forcing = teacher_forcing
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Move the entire data loader to the device. We're not working with that much data, so this
        # should actually speed things up.
        train_loader_on_device = [(input_tensor.to(self.device), target_tensor.to(self.device)) \
                                  for input_tensor, target_tensor in train_loader]

        best_loss = float("inf")  # Any loss will be better than this
        best_model_state = None

        if verbose:
            header = f"{'Epoch':<10}{'Train Loss':<15}"
            if test_loader is not None:
                header += f"{'Test Loss':<15}"

        training_losses = np.zeros(n_epochs)
        if test_loader is not None:
            test_losses = np.zeros(n_epochs)

        range_vals = range(n_epochs) if not verbose else trange(n_epochs)  # tqdm trange prints a nice little progress bar
        for epoch in range_vals:
            train_loss = self.train_epoch(train_loader_on_device)
            training_losses[epoch] = train_loss

            if test_loader is not None:
                test_loss = self.evaluate(test_loader)
                test_losses[epoch] = test_loss
            else:
                test_loss = None

            if optuna_trial is not None:
                optuna_trial.report(test_loss, step=epoch)
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            if verbose:
                msg = f"{epoch+1:<10}{train_loss:<15.5f}"
                if test_loss is not None:
                    msg += f"{test_loss:<15.5f}"
                # tqdm.write(msg)

            if test_loader is not None and test_loss < best_loss:
                best_loss = test_loss
                if save_best:
                    # Don't actually save the model yet, because that is slow. Also, we'd be
                    # having to save it almost every epoch at the start of training.
                    best_model_state = self.model.state_dict()

            if save_every is not None and (epoch+1) % save_every == 0:
                save_every_path = save_path.replace(".pt", f"_{epoch+1}.pt")
                torch.save(self.model.state_dict(), save_every_path)

        if save_best:
            torch.save(best_model_state, save_path)

        results = TrainingResults(self.model, train_loader, training_losses, test_loader, test_losses)
        return results

    def train_epoch(self, train_loader):
        """
        Train the model on the training data for one epoch.

        :param train_loader: torch.utils.data.DataLoader: The training data loader
        """
        total_loss = 0
        for input_tensor, target_tensor in train_loader:
            # Move the data to the device
            input_tensor = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            # Backward pass
            self.optimizer.zero_grad()
            if self.teacher_forcing and torch.rand(1).item() < self.teacher_forcing_ratio:
                # Use teacher forcing
                output = self.model(input_tensor, target_tensor)
            else:
                # No teacher forcing
                output = self.model(input_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        total_loss /= len(train_loader)
        return total_loss

    def evaluate(self,
                 data_loader: torch.utils.data.DataLoader,
                 transformer=None,
                 loss_func=None) -> float:
        """
        Evaluate the model on the data loader.

        :param data_loader: torch.utils.data.DataLoader: The data loader
        :param transformer: sklearn.base.TransformerMixin: A trained transformer or transformer pipeline
                            to transform the target tensor before calculating the loss.
        :return: float: The average loss
        """
        input_tensor, target_tensor = data_loader.dataset.tensors
        input_tensor = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        if transformer is not None:
            output = transformer.inverse_transform(output)
            target_tensor = transformer.inverse_transform(target_tensor)

        if loss_func is not None:
            loss = loss_func(output, target_tensor)
        else:
            loss = self.criterion(output, target_tensor)

        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        return loss


class TrainingResults:
    """
    Provides a container for summarizing the outcomes of training the model. Also provides tools for
    limited visualization of the training process.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 training_losses: np.ndarray,
                 test_loader: torch.utils.data.DataLoader | None = None,
                 test_losses: np.ndarray | None = None):
        self.model = model
        self.train_loader = train_loader
        self.training_losses = training_losses
        self.test_loader = test_loader
        self.test_losses = test_losses

    def plot_losses(self):
        """
        Plot the training and test losses.
        """
        import matplotlib.pyplot as plt

        plt.plot(self.training_losses, label="Training Loss")
        if self.test_losses is not None:
            plt.plot(self.test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    @property
    def best_test_loss(self):
        if self.test_losses is None:
            return None
        return np.min(self.test_losses)
