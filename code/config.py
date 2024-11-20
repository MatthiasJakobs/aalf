DATASET_HYPERPARAMETERS = {
    'australian_electricity_demand': {'freq': '30min', 'L': 48 },
    'pedestrian_counts': {'freq': '1h', 'L': 24 },
    'nn5_daily_nomissing': {'freq': '1d', 'L': 14 },
    'weather': {'freq': '1d', 'L': 14 },
    'kdd_cup_nomissing': {'freq': '1h', 'L': 24},
    'bitcoin_nomissing': {'freq': '1d', 'L': 14},
}

DEEPAR_HYPERPARAMETERS = {
    'australian_electricity_demand': {'num_layers': 1, 'hidden_size': 200, 'max_epochs':100, 'limit_train_batches': 1024, 'batch_size': 256, 'dropout': 0 },
    'weather': {'num_layers': 1, 'hidden_size': 25, 'max_epochs':100, 'limit_train_batches': 10_000, 'batch_size': 256, 'dropout': 0 },
    'nn5_daily_nomissing': {'num_layers': 1, 'hidden_size': 25, 'max_epochs':100, 'limit_train_batches': None, 'batch_size': 256, 'dropout': 0 },
    'pedestrian_counts': {'num_layers': 1, 'hidden_size': 100, 'max_epochs':200, 'limit_train_batches': None, 'batch_size': 256, 'dropout': 0 },
    'kdd_cup_nomissing': {'num_layers': 1, 'hidden_size': 25, 'max_epochs':100, 'limit_train_batches': None, 'batch_size': 256, 'dropout': 0, 'show_progress': True },
    'bitcoin_nomissing': {'num_layers': 1, 'hidden_size': 25, 'max_epochs':100, 'limit_train_batches': None, 'batch_size': 256, 'dropout': 0},
}

FCN_HYPERPARAMETERS = {
    'australian_electricity_demand': {'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': 1024},
    'nn5_daily_nomissing': {'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': None},
    'weather': {'hidden_size': 64, 'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': 10_000},
    'pedestrian_counts': {'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': None},
    'kdd_cup_nomissing': {'max_epochs': 100, 'learning_rate': 6e-4, 'batch_size': 256, 'limit_train_batches': None},
    'bitcoin_nomissing': {'max_epochs': 100, 'learning_rate': 6e-4, 'batch_size': 256, 'limit_train_batches': None},
}

ALL_DATASETS = ['australian_electricity_demand', 'nn5_daily_nomissing', 'weather', 'pedestrian_counts', 'kdd_cup_nomissing']

# Some maps for nicer rendered graphics
DS_MAP = {
    'pedestrian_counts': 'Pedestrian Counts',
    'nn5_daily_nomissing': 'NN5 (Daily)',
    'kdd_cup_nomissing': 'KDD Cup',
    'australian_electricity_demand': 'Aus. Elect. Demand',
    'weather': 'Weather',
}

MODEL_MAP = {
    'linear': 'Linear',
    'fcnn': 'FCNN',
    'deepar': 'DeepAR',
}

LOSS_MAP = {
    'rmse': 'RMSE',
    'smape': 'SMAPE',
}