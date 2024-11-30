from functools import partial
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from skopt import gp_minimize
from skopt import space

import torch
import torch.nn as nn
from src.model_training.lstm_model import LSTM

def hyperparameter_optimize(params, param_names, X_train, y_train, X_val, y_val):
    """
    Function to optimize the hyperparameters of a given model
    using Bayesian Optimization.
    Args:
        params : list
            list of hyperparameters to be optimized
        param_names : list
            list of hyperparameters names
        X_train : torch.Tensor
            training data
        y_train : torch.Tensor
            training labels
        X_val : torch.Tensor
            validation data
        y_val : torch.Tensor
            validation labels
    Returns:
        dict
            dictionary with the optimized hyperparameters
    """
    # convert params to dictionary
    params = dict(zip(param_names, params))
    
    hidden_dim = params["hidden_dim"]
    num_layers = params["num_layers"]
    dropout = params["dropout"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    #print(type(hidden_dim))
    hidden_dim = int(hidden_dim)
    input_dim = X_train.shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device).float()
    y_train = y_train.to(device).float()
    X_val = X_val.to(device).float()
    y_val = y_val.to(device).float()

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers, dropout=dropout)
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val)
        val_loss = loss_fn(y_pred, y_val)

    torch.cuda.empty_cache()

    return val_loss.item()


# Function to find the optimal hyperparameters
def find_optimal_params(X_train_torch, y_train_torch, X_val_torch, y_val_torch):
    '''
    Hyperparameters:
    hidden_dim: Dimension of hidden units in the LSTM layer
    num_layers: Number of LSTM layers
    dropout: Dropout rate of the LSTM layer
    lr: Learning rate of the optimizer
    weight_decay: Weight decay of the optimizer (L2 regularization)
    '''

    param_space = [
        space.Integer(10, 50, name="hidden_dim"),
        space.Integer(1, 3, name="num_layers"),
        space.Real(0, 0.5, prior="uniform", name="dropout"),
        space.Real(0.0001, 0.1, prior="log-uniform", name="lr"),
        space.Real(1e-6, 0.01, prior="log-uniform", name="weight_decay")
    ]

    param_names = ["hidden_dim", "num_layers", "dropout", "lr", "weight_decay"]

    optimization_function = partial(
        hyperparameter_optimize,
        param_names=param_names,
        X_train=X_train_torch,
        y_train=y_train_torch,
        X_val=X_val_torch,
        y_val=y_val_torch
    )

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
    )

    best_params = dict(zip(param_names, result.x))
    #print("Best Parameters:", best_params)
    torch.cuda.empty_cache()
    return best_params
