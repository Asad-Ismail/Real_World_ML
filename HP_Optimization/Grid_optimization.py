import itertools
from scipy.stats import uniform, loguniform

# define the model
def my_model(param1, param2, param3):
    # your model implementation here
    return model_score

# define the hyperparameter space
param_space = {
    'param1': [0.1, 0.5, 1.0],
    'param2': [10, 20, 30],
    'param3': loguniform(1e-6, 1.0)
}

# generate all combinations of hyperparameters
param_grid = []
for values in itertools.product(*param_space.values()):
    params = dict(zip(param_space.keys(), values))
    if isinstance(params['param3'], float):
        param_grid.append(params)
    else:
        param_grid += [{**params, 'param3': p3} for p3 in params['param3'].rvs(5)]

# perform grid search
best_score = float('-inf')
best_params = None
for params in param_grid:
    score = my_model(**params)
    if score > best_score:
        best_score = score
        best_params = params

# print the best parameters and score
print("Best parameters: ", best_params)
print("Best score: ", best_score)
