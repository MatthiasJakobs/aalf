DATASET_HYPERPARAMETERS = {
    'australian_electricity_demand': {'freq': '30min', 'L': 48 },
    'pedestrian_counts': {'freq': '1h', 'L': 24 },
    'nn5_daily_nomissing': {'freq': '1d', 'L': 14 },
    'weather': {'freq': '1d', 'L': 14 },
    'kdd_cup_nomissing': {'freq': '1h', 'L': 24},
}

DEEPAR_HYPERPARAMETERS = {
    'australian_electricity_demand': {'num_layers': 1, 'hidden_size': 200, 'max_epochs':100, 'limit_train_batches': 200, 'batch_size': 200, 'dropout': 0 },
    'weather': {'num_layers': 1, 'hidden_size': 25, 'max_epochs':100, 'limit_train_batches': None, 'batch_size': 200, 'dropout': 0 },
    'nn5_daily_nomissing': {'num_layers': 1, 'hidden_size': 25, 'max_epochs':100, 'limit_train_batches': None, 'batch_size': 200, 'dropout': 0 },
    'pedestrian_counts': {'num_layers': 1, 'hidden_size': 100, 'max_epochs':200, 'limit_train_batches': 200, 'batch_size': 200, 'dropout': 0 },
    'kdd_cup_nomissing': {'num_layers': 1, 'hidden_size': 100, 'max_epochs':100, 'limit_train_batches': None, 'batch_size': 200, 'dropout': 0, 'show_progress': True },
}

FCN_HYPERPARAMETERS = {
    'australian_electricity_demand': {'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': 200},
    'nn5_daily_nomissing': {'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': None},
    'weather': {'hidden_size': 64, 'max_epochs': 10, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': None},
    'pedestrian_counts': {'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': 200},
    'kdd_cup_nomissing': {'hidden_size': 64, 'max_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 256, 'limit_train_batches': None},
}