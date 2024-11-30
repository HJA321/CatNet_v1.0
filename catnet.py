import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess
import random

from src.model_training.lstm_model import LSTM
from src.model_training.train import train_lstm
from src.model_training.hyperparameter_tuning import hyperparameter_optimize, find_optimal_params
from src.factors.factor_processing import set_first_column_as_index
from src.FDR.shap_values import calc_shap_values
from FDR.kernel_dependence import *

# Load the data
def load_data(stock_code: str) -> pd.DataFrame:
    """
    Function to load the data.
    Args:
        stock_code : str
            stock code
    Returns:
        pd.DataFrame
            data
    """
    stock_data = pd.read_csv(os.path.join('/Users/hanjiaan/Documents/GitHub/CatNet/data/train_data_new', f'{stock_code}.csv'))
    stock_data = set_first_column_as_index(stock_data)
    stock_data.index = pd.to_datetime(stock_data.index)

    return stock_data

def preprocess_data(data: pd.DataFrame, train_end_date: str, lookback: int = 60):
    """
    Function to preprocess scaled training data.
    Args:
        data : pd.DataFrame
            data
        train_end_date : str
            end date of the training data
        lookback : int
            number of lookback days
    Returns:
        train_data_df : pd.DataFrame
            normalized training data
        train_label : np.array
            normalized training labels
        test_data_df : pd.DataFrame
            normalized test data
        test_label : np.array
            normalized test labels
        scaler : MinMaxScaler
            scaler that was used to normalize the data
        train_dates : pd.DatetimeIndex
            training dates
        test_dates : pd.DatetimeIndex
            test dates
    """
    features = list(data.columns)
    target = 'close'

    train_data = data.loc[data.index <= train_end_date]
    test_data = data.loc[data.index > train_end_date]
    train_label = train_data[target]
    test_label = test_data[target]
    train_dates = train_data.index
    test_dates = test_data.index

    # Normalize the data
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data).astype(np.float32)
    test_data = scaler.transform(test_data).astype(np.float32)
    train_label = scaler.fit_transform(train_label.values.reshape(-1, 1)).astype(np.float32)
    test_label = scaler.transform(test_label.values.reshape(-1, 1)).astype(np.float32)

    train_data_df = pd.DataFrame(train_data, columns=features, index=train_dates)
    test_data_df = pd.DataFrame(test_data, columns=features, index=test_dates)

    return train_data_df, train_label, test_data_df, test_label, scaler, train_dates, test_dates

def prepare_lstm_data(train_data_df: pd.DataFrame, train_label: np.array, test_data_df: pd.DataFrame, 
                      test_label: np.array, lookback: int = 60):
    """
    Function to prepare the data for LSTM training.
    Args:
        train_data_df : pd.DataFrame
            normalized training data
        train_label : np.array
            normalized training labels
        test_data_df : pd.DataFrame
            normalized test data
        test_label : np.array
            normalized test labels
        scaler : MinMaxScaler
            scaler used to normalize the data
        lookback : int
            number of lookback days
    Returns:
        train_data : torch.Tensor
            training data for LSTM
        train_label : torch.Tensor
            training labels for LSTM
        test_data : torch.Tensor
            test data for LSTM
        test_label : torch.Tensor
            test labels for LSTM
        scaler : MinMaxScaler
            scaler used to normalize the data
    """

    def load_data_X(data_X: pd.DataFrame, lookback_days = lookback):
        data_X_new = []
        for i in range(lookback_days, len(data_X)):
            # X is the past 60 days of data
            data_X_new.append(data_X.iloc[i-lookback_days:i].values)
        return np.array(data_X_new)
    
    X_train = load_data_X(train_data_df, lookback)
    all_data = pd.concat([train_data_df, test_data_df])
    test_size = len(test_data_df)
    X_test = load_data_X(all_data.iloc[-(test_size + lookback):], lookback)
    y_train = np.array(train_label[lookback:])
    y_test = np.array(test_label)

    train_data = torch.from_numpy(X_train).float()
    train_label = torch.from_numpy(y_train).float()
    test_data = torch.from_numpy(X_test).float()
    test_label = torch.from_numpy(y_test).float()

    return train_data, train_label, test_data, test_label

def train_lstm_model(train_data: torch.Tensor, train_label: torch.Tensor, test_data: torch.Tensor, test_label: torch.Tensor, num_epochs=300):
    """
    Function to train an LSTM model with hyperparameter tuning.
    Args:
        train_data : torch.Tensor
            training data for LSTM
        train_label : torch.Tensor
            training labels for LSTM
        test_data : torch.Tensor
            test data for LSTM
        test_label : torch.Tensor
            test labels for LSTM
        num_epochs : int
            number of epochs
    Returns:
        model : torch.nn.Module
            trained model
    """

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device).float()
    train_label = train_label.to(device).float()
    test_data = test_data.to(device).float()
    test_label = test_label.to(device).float()

    best_params = find_optimal_params(train_data, train_label, test_data, test_label)

    print(f"Best hyperparameters: {best_params}")
    input_dim = train_data.shape[2]
    output_dim = 1
    model = LSTM(input_dim=input_dim, hidden_dim=int(best_params['hidden_dim']), output_dim=output_dim,
                  num_layers=int(best_params['num_layers']), dropout=best_params['dropout'])
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    train_lstm(model, inputs=train_data, labels=train_label, num_epochs=num_epochs, criterion=criterion, optimizer=optimizer)

    torch.cuda.empty_cache()

    return model, optimizer

def train_and_test_score(model: nn.Module, scaler: MinMaxScaler, train_data: torch.Tensor, train_label: torch.Tensor, test_data: torch.Tensor, test_label: torch.Tensor):
    """
    Function to evaluate the model on the test data.
    Args:
        model : torch.nn.Module
            trained model
        scaler : MinMaxScaler
            scaler used to normalize the data
        train_data : torch.Tensor
            training data
        train_label : torch.Tensor
            training labels
        test_data : torch.Tensor
            test data
        test_label : torch.Tensor
            test labels
    Returns:
        train_score : float
            MSE score on the training data
        test_score : float
            MSE score on the test data
    """

    train_label_pred = model(train_data)
    test_label_pred = model(test_data)
    train_label_inv = scaler.inverse_transform(train_label.detach().numpy().reshape(-1, 1))
    test_label_inv = scaler.inverse_transform(test_label.detach().numpy().reshape(-1, 1))
    train_label_pred_inv = scaler.inverse_transform(train_label_pred.detach().numpy())
    test_label_pred_inv = scaler.inverse_transform(test_label_pred.detach().numpy())

    train_score = mean_squared_error(train_label_inv, train_label_pred_inv)
    test_score = mean_squared_error(test_label_inv, test_label_pred_inv)

    return train_score, test_score

def plot_predictions(model: nn.Module, scaler: MinMaxScaler, train_data: torch.Tensor,
                      train_label: torch.Tensor, test_data: torch.Tensor, test_label: torch.Tensor,
                      train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex, sample_days: int, stock_code: str):
    """
    Function to plot the predictions of the model.
    Args:
        model : torch.nn.Module
            trained model
        scaler : MinMaxScaler
            scaler used to normalize the data
        train_data : torch.Tensor
            training data
        train_label : torch.Tensor
            training labels
        test_data : torch.Tensor
            test data
        test_label : torch.Tensor
            test labels
        train_dates : pd.DatetimeIndex
            training dates
        test_dates : pd.DatetimeIndex
            test dates
        sample_days : int
            number of days for training data to plot
        stock_code : str
            stock code
    Returns:
        None
    """

    train_label_pred = model(train_data)
    test_label_pred = model(test_data)
    train_label_inv = scaler.inverse_transform(train_label.detach().numpy().reshape(-1, 1))
    test_label_inv = scaler.inverse_transform(test_label.detach().numpy().reshape(-1, 1))
    train_label_pred_inv = scaler.inverse_transform(train_label_pred.detach().numpy())
    test_label_pred_inv = scaler.inverse_transform(test_label_pred.detach().numpy())

    train_actual_color = 'cornflowerblue'
    train_pred_color = 'lightblue'
    test_actual_color = 'salmon'
    test_pred_color = 'lightcoral'

    plt.figure(figsize=(18,6))
    plt.plot(train_dates[-sample_days:], train_label_inv[-sample_days:], label="Training Data", color=train_actual_color)
    plt.plot(train_dates[-sample_days:], train_label_pred_inv[-sample_days:], label="Training Predictions", linewidth=1, color=train_pred_color)

    plt.plot(test_dates, test_label_inv, label="Test Data", color=test_actual_color)
    plt.plot(test_dates, test_label_pred_inv, label="Test Predictions", linewidth=1, color=test_pred_color)

    plt.title(f"Stock Price Prediction for {stock_code} using LSTM")
    plt.xlabel("Time")
    plt.ylabel("Stock Price (USD)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(color="lightgray")

    plt.show()


"""
Inputs: 
    model: the pre-trained LSTM model
    optimizer: the optimizer used to train the model
    data: dataframe storing the features (normalized)
    y: the response vector
    q: the False Discovery Rate level
Outputs: the set of the index of the significant features
"""
def CatNet(model: nn.Module, optimizer:optim.Optimizer, data:pd.DataFrame, y, q, lookback = 60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    features = data.columns
    data_index = data.index
    #data = data.to_numpy()
    features_dict = {i: features[i] for i in range(len(features))}

    #p is the number of features, n is the number of samples
    n, p = data.shape
    noise = np.random.normal(0, 1, size = (n, p))

    mirror_stats = {}

    def load_data_X(data_X: pd.DataFrame, lookback_days = lookback):
        data_X_new = []
        for i in range(lookback_days, len(data_X)):
            # X is the past 60 days of data
            data_X_new.append(data_X.iloc[i-lookback_days:i].values)
        return np.array(data_X_new)
    
    data_train = load_data_X(data, lookback)
    data_train = torch.from_numpy(data_train).type(torch.Tensor).to(device)
    original_shap_values = calc_shap_values(model, data_train, features)



    plot_path = "/Users/hanjiaan/Documents/GitHub/CatNet/plots"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    else:
        for file in os.listdir(plot_path):
            os.remove(os.path.join(plot_path, file))

    def plot_shap_values(Xplus_shap, Xminus_shap, X_shap, feature_name, mirrored_data):
        feature_plus_shap_values = Xplus_shap

        '''
        plt.figure(figsize=(18,6))
        plt.plot(data_index[lookback:,], feature_plus_shap_values, label=f"{feature_name}_plus", color='cornflowerblue')
        plt.title(f"{feature_name}_plus SHAP Values in time series (t)")
        plt.xlabel("Time")
        plt.ylabel("SHAP Value")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        #plt.legend()
        plt.grid(color="lightgray")
        save_path = f"plots/shap_values_{feature_name}_plus.png"
        plt.savefig(save_path)
        '''


        feature_minus_shap_values = Xminus_shap
        '''
        plt.figure(figsize=(18,6))
        plt.plot(data_index[lookback:,], feature_minus_shap_values, label=f"{feature_name}_minus", color='salmon')
        plt.title(f"{feature_name}_minus SHAP Values in time series(t)")
        plt.xlabel("Time")
        plt.ylabel("SHAP Value")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        #plt.legend()
        plt.grid(color="lightgray")
        save_path = f"plots/shap_values_{feature_name}_minus.png"
        plt.savefig(save_path)
        '''

        plt.figure(figsize=(18,6))
        plt.scatter(mirrored_data[f"{feature}_plus"].values[lookback:], feature_plus_shap_values, label=f"{feature_name}_plus", color='cornflowerblue')
        plt.title(f"{feature_name}_plus SHAP Values vs {feature_name}_plus Value (x)")
        plt.xlabel(f"{feature_name} Value")
        plt.ylabel("SHAP Value")
        plt.grid(color="lightgray")
        save_path = f"plots/shap_values_{feature_name}_plus_scatter.png"
        plt.savefig(save_path)

        plt.figure(figsize=(18,6))
        plt.scatter(mirrored_data[f"{feature}_minus"].values[lookback:], feature_minus_shap_values, label=f"{feature_name}_minus", color='salmon')
        plt.title(f"{feature_name}_minus SHAP Values vs {feature_name}_minus Value (x)")
        plt.xlabel(f"{feature_name} Value")
        plt.ylabel("SHAP Value")
        plt.grid(color="lightgray")
        save_path = f"plots/shap_values_{feature_name}_minus_scatter.png"
        
        plt.savefig(save_path)

        feature_original_shap_values = X_shap

        plt.figure(figsize=(18,6))
        plt.scatter(data[f"{feature_name}"].values[lookback:], feature_original_shap_values, label=f"{feature_name}", color='lightcoral')
        plt.title(f"{feature_name} original SHAP Values vs {feature_name} Value (x)")
        plt.xlabel(f"{feature_name} Value")
        plt.ylabel("SHAP Value")
        plt.grid(color="lightgray")
        save_path = f"plots/shap_values_{feature_name}_scatter.png"
        plt.savefig(save_path)


    #Generate the mirrored data
    for j in range(p):

        xj = data[features[j]]
        xj = xj.to_numpy()
        zj = noise[:, j]
        cj = find_cj(data, xj, zj, j, lookback = lookback)
        print(f"Optimal cj for feature {features[j]}: {cj}")

        mirrored_data = data.drop(columns = features[j])
        Xplus = pd.Series(xj + cj * zj, name = f"{features[j]}_plus", index = data_index)
        Xminus = pd.Series(xj - cj * zj, name = f"{features[j]}_minus", index = data_index)
        #Xplus = pd.Series(xj, name = f"{features[j]}_plus", index = data_index)
        #Xminus = pd.Series(xj, name = f"{features[j]}_minus", index = data_index)


        mirrored_data = pd.concat([mirrored_data, Xplus, Xminus], axis = 1)
        # Set the names of the new columns
        new_features = list(features.drop(features[j])) + [f"{features[j]}_plus", f"{features[j]}_minus"]
        mirrored_data.columns = new_features

        # mirrored_data = torch.tensor(mirrored_data, dtype = torch.float32)
        #y = torch.tensor(y, dtype = torch.float32).view(-1, 1)
        mirrored_data_train = load_data_X(mirrored_data, lookback)
        mirrored_data_train = torch.from_numpy(mirrored_data_train).type(torch.Tensor).to(device)
        y_train = np.array(y[lookback:])
        y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)

        input_size = p + 1
        hidden_size = model.lstm.hidden_size
        output_size = 1
        num_layers = model.lstm.num_layers
        dropout = model.lstm.dropout
        lr = optimizer.param_groups[0]['lr']
        weight_decay = optimizer.param_groups[0]['weight_decay']

        new_model = LSTM(input_size, hidden_size, num_layers, output_size, dropout)
        new_model.to(device)
        criterion = nn.MSELoss()
        new_optimizer = optim.Adam(new_model.parameters(), lr = lr, weight_decay = weight_decay)

        train_lstm(new_model, mirrored_data_train, y_train, 100, criterion, new_optimizer)

        # Use the SHAP values to calculate the importance of each feature
        #print(f"Calculating SHAP values for feature {j}...")
        # print(f"Size of mirrored data: {mirrored_data_train.shape}")
        shap_values = calc_shap_values(new_model, mirrored_data_train, new_features)

        Xplus_shap = shap_values[f"{features[j]}_plus"]
        Xminus_shap = shap_values[f"{features[j]}_minus"]
        X_shap = original_shap_values[features[j]]

        Xplus_importance = np.zeros(n - lookback)
        Xminus_importance = np.zeros(n - lookback)

        # Calculate the gradient of the lowess fit of the SHAP values with respect to the feature values
        # Take the sum of the gradients (path gradient) as the feature importance
        # for feature in features:
        feature = features[j]
        x = data[feature].values[lookback:]
        x_plus = mirrored_data[f"{feature}_plus"].values[lookback:]
        x_minus = mirrored_data[f"{feature}_minus"].values[lookback:]
        shap_plus = Xplus_shap
        shap_minus = Xminus_shap
        #print(shap_plus.shape, x.shape)
        lowess_plus = lowess(shap_plus, x_plus, frac = 0.3)
        lowess_minus = lowess(shap_minus, x_minus, frac = 0.3)
        X_plus_lowess, Y_plus_lowess = zip(*lowess_plus)
        X_minus_lowess, Y_minus_lowess = zip(*lowess_minus)
        Xminus_importance = np.gradient(Y_minus_lowess, X_minus_lowess)
        Xplus_importance = np.gradient(Y_plus_lowess, X_plus_lowess)
        #Xplus_importance += Xplus_grad_wrt_x
        #Xminus_importance += Xminus_grad_wrt_x

        # Calculate the L1 norm of the feature importance vector
        L_xj_plus = np.abs(Xplus_importance).sum()
        L_xj_minus = np.abs(Xminus_importance).sum()

        # Calculate the mirror statistics: sign(Corr(Xplus_importance, Xminus_importance)) * max(L_xj_plus, L_xj_minus)
        # corr = np.corrcoef(Xplus_importance, Xminus_importance)[0, 1]
        #print(f"Correlation between Xplus and Xminus importance: {corr}")
        # mirror_stat = np.sign(corr) * max(L_xj_plus, L_xj_minus)
        # sign_function = np.sign(np.sum(Xplus_importance * Xminus_importance))
        sign_function = np.dot(Xplus_importance, Xminus_importance) / (np.linalg.norm(Xplus_importance) * np.linalg.norm(Xminus_importance))
        mirror_stat = sign_function * max(L_xj_plus, L_xj_minus)

        print(f"Mirror Statistic for feature {features[j]}: {mirror_stat}")
        mirror_stats[f"{features[j]}"] = mirror_stat

        # Plot the SHAP values of the mirrored data
        plot_shap_values(Xplus_shap, Xminus_shap, X_shap, features[j], mirrored_data)



    #For a designated FDR level q, calculate the cutoff Tq
    sorted_mirror_stats = sorted(mirror_stats.items(), key = lambda x: x[1], reverse = False)
    cutoff = 0
    print(f"Sorted Mirror Stats: {sorted_mirror_stats}")

    for feature_name, mirror_stat in sorted_mirror_stats:
        if mirror_stat <= 0:
            continue
        num_j_less = np.sum(m[1] <= -mirror_stat for m in sorted_mirror_stats)
        num_j_more = np.sum(m[1] >= mirror_stat for m in sorted_mirror_stats)
        if num_j_more == 0:
            raise ValueError("No feature is selected.")
        else:
            ratio = (num_j_less + 1) / num_j_more
            if ratio <= q:
                cutoff = mirror_stat
                break
    

    #Get the index of the significant features
    selected_features_indices = np.where(np.array(list(mirror_stats.values())) >= cutoff)[0]
    selected_features = {features_dict[i]: mirror_stats[features_dict[i]] for i in selected_features_indices}

    return selected_features, mirror_stats, cutoff


def S_CatNet(model: nn.Module, optimizer:optim.Optimizer, data:pd.DataFrame, y, q, lookback = 60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    features = data.columns
    data_index = data.index
    #data = data.to_numpy()
    features_dict = {i: features[i] for i in range(len(features))}

    #p is the number of features, n is the number of samples
    n, p = data.shape
    noise = np.random.normal(0, 1, size = (n, p))

    mirror_stats = {}

    def load_data_X(data_X: pd.DataFrame, lookback_days = lookback):
        data_X_new = []
        for i in range(lookback_days, len(data_X)):
            # X is the past 60 days of data
            data_X_new.append(data_X.iloc[i-lookback_days:i].values)
        return np.array(data_X_new)
    
    data_train = load_data_X(data, lookback)
    data_train = torch.from_numpy(data_train).type(torch.Tensor).to(device)
    # original_shap_values = calc_shap_values(model, data_train, features)


    mirrored_data = data.copy()
    for j in range(p):
        xj = data[features[j]]
        xj = xj.to_numpy()
        zj = noise[:, j]
        cj = find_cj(data, xj, zj, j, lookback = lookback)

        #mirrored_data = data.drop(columns = features[j])
        Xplus = pd.Series(xj + cj * zj, name = f"{features[j]}_plus", index = data_index)
        Xminus = pd.Series(xj - cj * zj, name = f"{features[j]}_minus", index = data_index)
        # Xplus = pd.Series(xj, name = f"{features[j]}_plus", index = data_index)
        # Xminus = pd.Series(xj, name = f"{features[j]}_minus", index = data_index)

        mirrored_data = pd.concat([mirrored_data, Xplus, Xminus], axis = 1)
        

    # Drop the original features
    mirrored_data = mirrored_data.drop(columns = features)
    print(mirrored_data.columns)
    print(mirrored_data.shape)
    
    new_features = mirrored_data.columns
    print(new_features)
    
    mirrored_data_train = load_data_X(mirrored_data, lookback)

    mirrored_data_train = torch.from_numpy(mirrored_data_train).type(torch.Tensor).to(device)
    y_train = np.array(y[lookback:])
    y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)

    input_size = 2*p
    hidden_size = model.lstm.hidden_size
    output_size = 1
    num_layers = model.lstm.num_layers
    dropout = model.lstm.dropout
    lr = optimizer.param_groups[0]['lr']
    weight_decay = optimizer.param_groups[0]['weight_decay']

    new_model = LSTM(input_size, hidden_size, num_layers, output_size, dropout)
    new_model.to(device)
    criterion = nn.MSELoss()
    new_optimizer = optim.Adam(new_model.parameters(), lr = lr, weight_decay = weight_decay)

    train_lstm(new_model, mirrored_data_train, y_train, 100, criterion, new_optimizer)

    shap_values = calc_shap_values(new_model, mirrored_data_train, new_features)

    '''
    plot_path = "/Users/hanjiaan/Documents/GitHub/CatNet/plots"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    else:
        for file in os.listdir(plot_path):
            os.remove(os.path.join(plot_path, file))

    def plot_shap_values(Xplus_shap, Xminus_shap, X_shap, feature_name, mirrored_data):
        
        feature_plus_shap_values = Xplus_shap
        feature_minus_shap_values = Xminus_shap

        plt.figure(figsize=(18,6))
        plt.scatter(mirrored_data[f"{feature}_plus"].values[lookback:], feature_plus_shap_values, label=f"{feature_name}_plus", color='cornflowerblue')
        plt.title(f"{feature_name}_plus SHAP Values vs {feature_name}_plus Value (x)")
        plt.xlabel(f"{feature_name} Value")
        plt.ylabel("SHAP Value")
        plt.grid(color="lightgray")
        save_path = f"plots/shap_values_{feature_name}_plus_scatter.png"
        plt.savefig(save_path)

        plt.figure(figsize=(18,6))
        plt.scatter(mirrored_data[f"{feature}_minus"].values[lookback:], feature_minus_shap_values, label=f"{feature_name}_minus", color='salmon')
        plt.title(f"{feature_name}_minus SHAP Values vs {feature_name}_minus Value (x)")
        plt.xlabel(f"{feature_name} Value")
        plt.ylabel("SHAP Value")
        plt.grid(color="lightgray")
        save_path = f"plots/shap_values_{feature_name}_minus_scatter.png"
        
        plt.savefig(save_path)

        feature_original_shap_values = X_shap

        plt.figure(figsize=(18,6))
        plt.scatter(data[f"{feature_name}"].values[lookback:], feature_original_shap_values, label=f"{feature_name}", color='lightcoral')
        plt.title(f"{feature_name} original SHAP Values vs {feature_name} Value (x)")
        plt.xlabel(f"{feature_name} Value")
        plt.ylabel("SHAP Value")
        plt.grid(color="lightgray")
        save_path = f"plots/shap_values_{feature_name}_scatter.png"
        plt.savefig(save_path)
    '''

    for j in range(p):
        Xplus_shap = shap_values[f"{features[j]}_plus"]
        Xminus_shap = shap_values[f"{features[j]}_minus"]
        # X_shap = original_shap_values[features[j]]

        Xplus_importance = np.zeros(n - lookback)
        Xminus_importance = np.zeros(n - lookback)

        feature = features[j]
        # x = data[feature].values[lookback:]
        x_plus = mirrored_data[f"{feature}_plus"].values[lookback:]
        x_minus = mirrored_data[f"{feature}_minus"].values[lookback:]
        shap_plus = Xplus_shap
        shap_minus = Xminus_shap
        #print(shap_plus.shape, x.shape)
        lowess_plus = lowess(shap_plus, x_plus, frac = 0.3)
        lowess_minus = lowess(shap_minus, x_minus, frac = 0.3)
        X_plus_lowess, Y_plus_lowess = zip(*lowess_plus)
        X_minus_lowess, Y_minus_lowess = zip(*lowess_minus)
        Xminus_importance = np.gradient(Y_minus_lowess, X_minus_lowess)
        Xplus_importance = np.gradient(Y_plus_lowess, X_plus_lowess)

        L_xj_plus = np.abs(Xplus_importance).sum()
        L_xj_minus = np.abs(Xminus_importance).sum()

        # Calculate the mirror statistics: sign(Corr(Xplus_importance, Xminus_importance)) * max(L_xj_plus, L_xj_minus)
        #corr = np.corrcoef(Xplus_importance, Xminus_importance)[0, 1]
        #print(f"Correlation between Xplus and Xminus importance: {corr}")
        #mirror_stat = np.sign(corr) * max(L_xj_plus, L_xj_minus)
        #sign_function = np.sign(np.sum(Xplus_importance * Xminus_importance))
        sign_function = np.dot(Xplus_importance, Xminus_importance) / (np.linalg.norm(Xplus_importance) * np.linalg.norm(Xminus_importance))
        mirror_stat = sign_function * max(L_xj_plus, L_xj_minus)

        print(f"Mirror Statistic for feature {features[j]}: {mirror_stat}")
        mirror_stats[f"{features[j]}"] = mirror_stat

        # Plot the SHAP values of the mirrored data
        #plot_shap_values(Xplus_shap, Xminus_shap, X_shap, features[j], mirrored_data)

    #For a designated FDR level q, calculate the cutoff Tq
    sorted_mirror_stats = sorted(mirror_stats.items(), key = lambda x: x[1], reverse = False)
    cutoff = 0
    print(f"Sorted Mirror Stats: {sorted_mirror_stats}")

    for feature_name, mirror_stat in sorted_mirror_stats:
        if mirror_stat <= 0:
            continue
        num_j_less = np.sum(m[1] <= -mirror_stat for m in sorted_mirror_stats)
        num_j_more = np.sum(m[1] >= mirror_stat for m in sorted_mirror_stats)
        if num_j_more == 0:
            raise ValueError("No feature is selected.")
        else:
            ratio = (num_j_less + 1) / num_j_more
            if ratio <= q:
                cutoff = mirror_stat
                break
    

    #Get the index of the significant features
    selected_features_indices = np.where(np.array(list(mirror_stats.values())) >= cutoff)[0]
    selected_features = {features_dict[i]: mirror_stats[features_dict[i]] for i in selected_features_indices}

    return selected_features, mirror_stats, cutoff



