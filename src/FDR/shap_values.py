import shap
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
#from src.model_training.lstm_model import LSTM

def calc_shap_values(model: nn.Module, X, feature_names):
    """
    Function to calculate SHAP values for a given model.
    Args:
        model : torch.nn.Module
            trained model
        X : torch.Tensor
            input data
        feature_names : list
            list of feature names
    Returns:
        dict
            dictionary with the SHAP values
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X = X.to(device)

    model_was_training = model.training
    model.train()
    cudnn.enabled = False

    explainer = shap.DeepExplainer(model, X)
    shap_values = explainer.shap_values(X, check_additivity=False)

    if not model_was_training:
        model.eval()

    cudnn.enabled = True

    # shap_values = shap_values.values.squeeze() 
    lookback_sum = np.sum(shap_values, axis=1).squeeze()

    # Take the weighted average of the SHAP values of the last 10 lookback days as the feature importance
    #time_step_importance_weights = time_step_importance_mean[-10:] / time_step_importance_mean[-10:].sum()
    feature_importance = lookback_sum.T

    return dict(zip(feature_names, feature_importance))