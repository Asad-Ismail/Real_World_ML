import random
from sklearn.model_selection import cross_val_score

def my_model(param1, param2, param3):
    return model_score

# define the hyperparameter space
param_space = {
    'param1': [0.1, 0.5, 1.0],
    'param2': [10, 20, 30],
    'param3': {'min': 0.0, 'max': 1.0, 'type': 'real'}
}

# define the search
n_iter = 20
best_score = float('-inf')
best_params = None
for i in range(n_iter):
    params = {
        'param1': random.choice(param_space['param1']),
        'param2': random.choice(param_space['param2']),
        'param3': random.uniform(param_space['param3']['min'], param_space['param3']['max'])
    }
    score = my_model(**params)
    if score > best_score:
        best_score = score
        best_params = params

# print the best parameters and score
print("Best parameters: ", best_params)
print("Best score: ", best_score)
