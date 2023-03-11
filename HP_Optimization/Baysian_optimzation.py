import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimization:
    def __init__(self, f, bounds, init_points=5, acq='ucb', kappa=2.576):
        """
        :param f: function to optimize
        :param bounds: bounds of the input variables [(min, max), (min, max), ...]
        :param init_points: number of initial random points
        :param acq: acquisition function to use (either 'ucb' or 'ei')
        :param kappa: parameter for ucb acquisition function
        """
        self.f = f
        self.bounds = bounds
        self.init_points = init_points
        self.acq = acq
        self.kappa = kappa
        self.X = []
        self.y = []
        
    def acquisition_function(self, X, model, y_best):
        """
        :param X: input points to evaluate the acquisition function
        :param model: surrogate model
        :param y_best: best observed value of f so far
        :return: value of the acquisition function at each point in X
        """
        if self.acq == 'ucb':
            mean, std = model.predict(X, return_std=True)
            return mean + self.kappa * std
        elif self.acq == 'ei':
            mean, std = model.predict(X, return_std=True)
            z = (y_best - mean) / std
            return (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)
        
    def optimize(self, n_iter):
        """
        :param n_iter: number of iterations
        :return: best observed value of f and corresponding input point
        """
        for i in range(self.init_points):
            x = [np.random.uniform(b[0], b[1]) for b in self.bounds]
            y = self.f(x)
            self.X.append(x)
            self.y.append(y)
        
        for i in range(n_iter):
            model = GaussianProcessRegressor()
            model.fit(self.X, self.y)
            y_best = np.min(self.y)
            x_best = self.X[np.argmin(self.y)]
            
            if self.acq == 'ucb':
                res = minimize(lambda x: -self.acquisition_function(x.reshape(1, -1), model, y_best),
                               x0=np.random.uniform(self.bounds[0][0], self.bounds[0][1], len(self.bounds)))
            elif self.acq == 'ei':
                res = minimize(lambda x: -self.acquisition_function(x.reshape(1, -1), model, y_best),
                               x0=np.random.uniform(self.bounds[0][0], self.bounds[0][1], len(self.bounds)))
            
            x_new = res.x.tolist()
            y_new = self.f(x_new)
            self.X.append(x_new)
            self.y.append(y_new)
            
        return x_best, y_best
