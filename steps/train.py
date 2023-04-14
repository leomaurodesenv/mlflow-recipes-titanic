"""
This module defines the following routines used by the 'train' step:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model recipe.
"""
from typing import Any, Dict

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torchmetrics
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class PytorchDataset(Dataset):
    """
    Pytorch dataset from pd.DataFrame values
    """
    def __init__(self, X, y):
        x = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(X, pd.DataFrame) else y
        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx], self.y_train[idx]


class PytorchBinaryModel(pl.LightningModule):
    """
    Binary Module using LightningModule
    """
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.loss = nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        # it is independent of forward
        x, y = batch
        pred = self.architecture(x)
        # print(pred)
        loss = self.loss(pred, y)
        # log step metric
        self.accuracy(pred, y)
        self.log('train_acc_step', self.accuracy)
        return loss

    def forward(self, x):
        pred = self.architecture(x)
        return pred

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class CustomClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom sklearn estimator using pytorch model
    """
    def __init__(self, input_size, batch_size=32):
        self.classes = [0, 1]
        self.input_size = input_size
        self.batch_size = batch_size
        self.architecture = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.base = PytorchBinaryModel(self.architecture)

    def fit(self, X, y):
        # Create dataloader
        dataset = PytorchDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        # Train the model
        trainer = pl.Trainer(max_epochs=50)
        trainer.fit(model=self.base, train_dataloaders=data_loader)

    def predict(self, X):
        x = X.values if isinstance(X, pd.DataFrame) else X
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.base(x)
        pred = pred.reshape(-1).detach().numpy().round().astype(np.int32)
        return pred


def estimator_fn(estimator_params: Dict[str, Any] = None) -> Any:
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    return CustomClassifier(input_size=16)
