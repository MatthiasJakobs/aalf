DATASET_HYPERPARAMETERS = {
    'australian_electricity_demand': {'freq': '30min', 'L': 48, 'S':48, 'fint': 'linear', 'fcomp': 'fcnn' },
    'pedestrian_counts': {'freq': '1h', 'L': 24 , 'S': 24, 'fint': 'linear', 'fcomp': 'deepar' },
    'nn5_daily_nomissing': {'freq': '1d', 'L': 14, 'S': 7, 'fint': 'linear', 'fcomp': 'cnn' },
    'weather': {'freq': '1d', 'L': 14, 'S': 7, 'fint': 'linear', 'fcomp': 'deepar' },
    'kdd_cup_nomissing': {'freq': '1h', 'L': 24, 'S': 24, 'fint': 'linear', 'fcomp': 'fcnn' },
    'solar_10_minutes':{'freq': '1h', 'L': 24, 'S': 24, 'fint': 'linear', 'fcomp': 'cnn' }
}

DEEPAR_HYPERPARAMETERS = {
    'australian_electricity_demand': {'num_layers': 1, 'hidden_size': 200, 'max_epochs':100, 'limit_train_batches': 1024, 'batch_size': 256, 'dropout': 0 },
    'weather': {'num_layers': 1, 'hidden_size': 25, 'max_epochs':100, 'limit_train_batches': 10_000, 'batch_size': 256, 'dropout': 0 },
    'nn5_daily_nomissing': {'num_layers': 1, 'hidden_size': 25, 'max_epochs':100, 'limit_train_batches': None, 'batch_size': 256, 'dropout': 0 },
    'pedestrian_counts': {'num_layers': 1, 'hidden_size': 100, 'max_epochs':200, 'limit_train_batches': None, 'batch_size': 256, 'dropout': 0 },
    'kdd_cup_nomissing': {'num_layers': 1, 'hidden_size': 16, 'max_epochs':20, 'limit_train_batches': None, 'batch_size': 64, 'dropout': 0},
    'solar_10_minutes': {'learning_rate': 6e-4, 'num_layers': 1, 'hidden_size': 16, 'max_epochs':10, 'limit_train_batches': 1024, 'batch_size': 64, 'dropout': 0},
}

FCN_HYPERPARAMETERS = {
    'australian_electricity_demand': {'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': 1024},
    'nn5_daily_nomissing': {'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': None},
    'weather': {'hidden_size': 64, 'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': 10_000},
    'pedestrian_counts': {'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': None},
    'kdd_cup_nomissing': {'max_epochs': 20, 'learning_rate': 6e-4, 'batch_size': 64, 'limit_train_batches': None},
    'solar_10_minutes': {'max_epochs': 10, 'learning_rate': 1e-3, 'batch_size': 64, 'limit_train_batches': None},
}

CNN_HYPERPARAMETERS = {
    'weather': {'n_hidden_channels': 64, 'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': 10_000},
    'australian_electricity_demand': {'n_hidden_channels': 32, 'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 64, 'limit_train_batches': 1024},
    'nn5_daily_nomissing': {'n_hidden_channels': 32, 'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 64, 'limit_train_batches': None},
    'pedestrian_counts': {'n_hidden_channels': 32, 'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': None},
    'kdd_cup_nomissing': {'n_hidden_channels': 32, 'max_epochs': 100, 'learning_rate': 6e-4, 'batch_size': 64, 'limit_train_batches': None},
    'solar_10_minutes': {'n_hidden_channels': 32, 'max_epochs': 10, 'learning_rate': 1e-3, 'batch_size': 64, 'limit_train_batches': None},
}

ALL_DATASETS = ['australian_electricity_demand', 'nn5_daily_nomissing', 'weather', 'pedestrian_counts', 'kdd_cup_nomissing', 'solar_10_minutes']

MULTIVARIATE_DATASETS = [
]

# Some maps for nicer rendered graphics
DS_MAP = {
    'pedestrian_counts': 'Pedestrian Counts',
    'nn5_daily_nomissing': 'NN5 (Daily)',
    'kdd_cup_nomissing': 'KDD Cup',
    'australian_electricity_demand': 'Aus. Electricity Demand',
    'weather': 'Weather',
    'solar_10_minutes': 'Solar',
}

MODEL_MAP = {
    'linear': 'AR',
    'fcnn': 'FCNN',
    'deepar': 'DeepAR',
    'cnn': 'CNN',
    'autoarima': 'AutoAR',
    'autoets': 'AutoETS',
    'lastvalue': 'LV',
    'meanvalue': 'MV',
}

LOSS_MAP = {
    'rmse': 'RMSE',
    'smape': 'SMAPE',
}
