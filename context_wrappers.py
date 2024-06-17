import torch
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class ContextPipeline(BaseEstimator, TransformerMixin):
    """
    Implements a pipeline that works with 3D arrays. All operations are performed along the
    last axis.
    """
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y, context, **kwargs):
        for name, step in self.steps[:-1]:
            X = step.fit_transform(X)
        self.steps[-1][1].fit(X, y, context, **kwargs)
        return self

    def predict(self, X, context):
        for name, step in self.steps[:-1]:
            X = step.transform(X)
        ypred = self.steps[-1][1].predict(X, context)
        return ypred

    def fit_predict(self, X, y, context):
        return self.fit(X, y, context).transform(X, context)

    def score(self, X, y, context):
        return self.steps[-1][1].score(X, y, context)

    @property
    def losses(self):
        return self.steps[-1][1].losses


class ContextTransformedTargetRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, regressor, transformer, context_transformer):
        self.regressor = regressor
        self.transformer = transformer
        self.context_transformer = context_transformer

    def fit(self, X, y, context, **fit_params):
        yt = self.transformer.fit_transform(y)
        contextt = self.context_transformer.fit_transform(context)
        return self.regressor.fit(X, yt, contextt, **fit_params)

    def predict(self, X, context):
        contextt = self.context_transformer.transform(context)
        ytpred = self.regressor.predict(X, contextt)
        ypred = self.transformer.inverse_transform(ytpred)
        return ypred

    def fit_predict(self, X, y, context):
        self.fit(X, y, context)
        return self.predict(X, context)

    def score(self, X, y, context):
        contextt = self.context_transformer.transform(context)
        yt = self.transformer.transform(y)
        return self.regressor.score(X, yt, contextt)

    @property
    def losses(self):
        return self.regressor.losses


class NeuralNetRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, model, optimizer, criterion, epochs=100, batch_size=32, shuffle=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.losses = None

    def fit(self, X, y, context, **kwargs):
        device = torch.cuda.is_available() or "cpu"
        self.model.to(device)

        X = torch.Tensor(X)
        y = torch.Tensor(y)
        context = torch.Tensor(context)
        dataset = torch.utils.data.TensorDataset(X, y, context)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        teacher_forcing = kwargs.get("teacher_forcing", False)
        if teacher_forcing:
            forcing_ratio = kwargs.get("teacher_forcing_ratio", 0.5)

        losses = torch.zeros(self.epochs, device=device, requires_grad=False)
        bar = tqdm(range(self.epochs))
        for i in bar:
            total_loss = 0
            for input_tensor, target_tensor, context_tensor in loader:
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)
                context_tensor = context_tensor.to(device)

                self.optimizer.zero_grad()
                if teacher_forcing and torch.rand(1).item() < forcing_ratio:
                    output = self.model(input_tensor, context_tensor, target_tensor)
                else:
                    output = self.model(input_tensor, context_tensor)
                loss = self.criterion(output, target_tensor)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            total_loss /= len(loader)
            bar.set_postfix({"loss": f"{total_loss:7.4f}"})
            losses[i] = total_loss

        self.losses = losses

        return self

    def predict(self, X, context):
        self.model.to("cpu")
        X = torch.Tensor(X)
        context = torch.Tensor(context)
        ypred = self.model(X, context)
        ypred = ypred.detach().numpy()
        return ypred

    def score(self, X, y, context):
        ypred = self.predict(X, context)
        score = self.criterion(ypred, y)
        return score
