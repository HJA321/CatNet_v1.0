import torch
import torch.nn as nn
import numpy as np


def train_lstm(model, inputs, labels, num_epochs, criterion = nn.MSELoss(), optimizer = torch.optim.Adam):
    """
    Function to train a given LSTM model.
    Args:
        model : torch.nn.Module
            LSTM model
        inputs : torch.Tensor
            input data
        num_epochs : int
            number of epochs
        labels : torch.Tensor
            labels
        criterion : torch.nn.Module
            loss function
        optimizer : torch.optim.Optimizer
            optimizer
    Returns:
        None
    """

    hist = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        hist[epoch] = loss.item()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch} loss: {loss.item()}')
        
    
    print(f"Finished Training. Final loss: {loss.item()}")
    
    return None