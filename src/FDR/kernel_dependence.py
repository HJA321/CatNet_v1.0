import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch import linalg as LA
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline, lagrange

from src.FDR.shap_values import calc_shap_values
from src.model_training.lstm_model import LSTM
from src.model_training.train import train_lstm

'''
def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma = 1.0) -> torch.Tensor:
    #Computes the Gaussian (RBF) Kernel between two vectors.
    return torch.exp(-LA.norm(x - y) ** 2 / (2 * sigma ** 2))
'''


#Computes the Gaussian (RBF) Kernel Matrix of the input data (vector)
def calculate_kernel_matrix(X_sample: np.ndarray, sigma = 1.0) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Using broadcasting to compute the pairwise squared Euclidean distances
    X_sample_sq = X_sample ** 2
    X_sample_sq = torch.tensor(X_sample_sq, dtype = torch.float32, device = device)
    X_sample = torch.tensor(X_sample, dtype = torch.float32, device = device)

    dist_matrix = -2 * torch.outer(X_sample, X_sample) + X_sample_sq.view(-1, 1) + X_sample_sq.view(1, -1)

    # Compute the RBF kernel matrix
    K = torch.exp(-dist_matrix / (2 * sigma ** 2))
    torch.cuda.empty_cache()
    return K



def center_matrix(A: torch.Tensor) -> torch.Tensor:
    """
    Centers the kernel matrix A using the formula H = I - 1/n * 11^T.
    The centered matrix is A_c = HAH = A - row_sum/n - col_sum/n + total_sum/n^2,
    where:
      - row_sum is the sum of each row of A (column vector of shape (n, 1)),
      - col_sum is the sum of each column of A (row vector of shape (1, n)),
      - total_sum is the sum of all elements in A (scalar).
    """
    n = A.shape[0]
   
    row_sum = torch.sum(A, axis = 1)
    col_sum = torch.sum(A, axis = 0)
    total_sum = torch.sum(A)
    
    A_centered = A - row_sum.view(-1, 1) / n - col_sum.view(1, -1) / n + total_sum / (n ** 2)
    torch.cuda.empty_cache()
    return A_centered

def kernel_dependence(cj: float, X: np.ndarray, xj: np.ndarray, zj: np.ndarray, j: int, sigma = 1.0) -> float:
    """
    Calculate the kernel-based conditional dependence measure I_j
    Inputs:
        cj: the coefficient c_j
        X: the feature matrix
        xj: the j-th feature
        zj: the noise vector
        j: the index of the feature
        sigma: the bandwidth parameter for the RBF kernel
    Outputs:
        I_j: the kernel-based conditional dependence measure
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Generate xj_plus and xj_minus
    xj_plus = xj + cj * zj
    xj_minus = xj - cj * zj
    n = len(xj)

    #Compute kernel matrices
    K_U = calculate_kernel_matrix(xj_plus, sigma)
    K_V = calculate_kernel_matrix(xj_minus, sigma)
    #X_minus_j = np.delete(X, j, axis = 1)
    #K_W = calculate_kernel_matrix(X_minus_j, sigma)

    #Center the kernel matrices
    K_U_centered = center_matrix(K_U)
    K_V_centered = center_matrix(K_V)

    #Compute I_j
    #I_j = (K_U_centered * K_V_centered * K_W).sum().item() / (n ** 2)
    K_U_centered = K_U_centered.cpu().numpy()
    K_V_centered = K_V_centered.cpu().numpy()

    I_j = np.trace(K_U_centered @ K_V_centered) / (n**2)

    torch.cuda.empty_cache()
    return I_j

def timeseries_kernel_dependence(cj: float, X: np.ndarray, xj: np.ndarray, zj: np.ndarray, j: int, lookback = 60, sigma = 1.0) -> float:
    """
    Calculate the weighted average of time-series kernel-based conditional dependence measure I_j
    Inputs:
        cj: the coefficient c_j
        X: the feature matrix
        xj: the j-th feature
        zj: the noise vector
        j: the index of the feature
        lookback: the number of lookback time steps in the LSTM model
        sigma: the bandwidth parameter for the RBF kernel
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Generate xj_plus and xj_minus
    xj_plus = xj + cj * zj
    xj_minus = xj - cj * zj
    n = len(xj)

    # Take the weights through empirical formula
    weights = np.exp((-1/10) * np.arange(lookback + 1))
    weights = weights / np.sum(weights)
    #weights = (1 / (lookback + 1)) * np.ones(lookback + 1)

    K_U = calculate_kernel_matrix(xj_plus, sigma)
    K_V = calculate_kernel_matrix(xj_minus, sigma)
    K_U_centered = center_matrix(K_U)
    K_V_centered = center_matrix(K_V)
    K_U_centered = K_U_centered.cpu().numpy()
    K_V_centered = K_V_centered.cpu().numpy()
    I_j = np.trace(K_U_centered @ K_V_centered) / (n ** 2)

    for tau in range(1,lookback+1):
        xj_plus_t = xj_plus[tau:]
        xj_minus_tau = xj_minus[:-tau]
        n_tau = len(xj_plus_t)
        K_U = calculate_kernel_matrix(xj_plus_t, sigma)
        K_V = calculate_kernel_matrix(xj_minus_tau, sigma)
        K_U_centered = center_matrix(K_U)
        K_V_centered = center_matrix(K_V)
        K_U_centered = K_U_centered.cpu().numpy()
        K_V_centered = K_V_centered.cpu().numpy()
        I_j_tau = np.trace(K_U_centered @ K_V_centered) / (n_tau ** 2)
        I_j += weights[tau] * I_j_tau

    torch.cuda.empty_cache()
    return I_j


# Compute the c_j which is the argmin of I_{K_j}(c) using monte carlo and interpolation
def find_cj(X, xj, zj, j, lookback=30, sigma=1.0):
    '''
    n = X.shape[0]
    sample_size = int(sample_fraction * n)
    indices = np.random.choice(n, sample_size, replace = False)
    X_sample = X[indices, :]
    xj = xj[indices]
    zj = zj[indices]
    '''

    c_values = np.linspace(0.01, 1, 50)
    I_kj_values = []
    for c in c_values:
        #I_kj = kernel_dependence(c, X, xj, zj, j, sigma)
        I_kj = timeseries_kernel_dependence(c, X, xj, zj, j, lookback=lookback, sigma = sigma)
        I_kj_values.append(I_kj)
    
    min_index = np.argmin(I_kj_values)
    c_min = c_values[min_index]

    window_size = 5
    start_index = max(0, min_index - window_size)
    end_index = min(len(c_values), min_index + window_size)

    c_window = c_values[start_index:end_index]
    I_kj_window = I_kj_values[start_index:end_index]

    spline = UnivariateSpline(c_window, I_kj_window, k = 3, s = 0)

    c_finer = np.linspace(c_window[0], c_window[-1], 100)
    I_kj_finer = spline(c_finer)
    min_index_finer = np.argmin(I_kj_finer)
    c_optimal = c_finer[min_index_finer]

    return c_optimal


