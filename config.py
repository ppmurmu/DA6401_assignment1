sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'val_accuracy'
    },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64]
        },
        'learning_rate': {
            'values': [0.001, 0.0001]
        },
        'neurons': {
            'values': [32, 64, 128]
        },
        'hidden_layers': {
            'values': [3, 4, 5]
        },
        'activation': {
            'values': ['sigmoid', 'tanh','relu' ]
        },
        'weight_init': {
            'values': ['xavier', 'random']
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
        },
        'loss': {
            'value': 'ce'
        },
        'epochs': {
            'values': [5,10]
        },
        'decay': {
            'values': [0, 0.0005,  0.5]
        },
        'dataset': {
            'value': 'fashion_mnist'
        }
    }
}