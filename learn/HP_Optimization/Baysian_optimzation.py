import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class BayesianOptimization:
    def __init__(self, f, bounds, init_points=5, acq='ei', kappa=2.576):
        self.f = f
        self.bounds = np.array(bounds)
        self.init_points = init_points
        self.acq = acq
        self.kappa = kappa
        self.X = []
        self.y = []
        
        # Define default kernel
        self.kernel = Matern(nu=2.5)
    
    def acquisition_function(self, X, model, y_best):
        mean, std = model.predict(X.reshape(-1, len(self.bounds)), return_std=True)
        
        if self.acq == 'ucb':  # Upper Confidence Bound
            return mean + self.kappa * std
            
        elif self.acq == 'ei':  # Expected Improvement
            # Avoid division by zero
            with np.errstate(divide='warn'):
                imp = y_best - mean
                Z = imp / (std + 1e-9)
                ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
                ei[std == 0.0] = 0.0
            return ei
            
        elif self.acq == 'pi':  # Probability of Improvement
            z = (mean - y_best - self.kappa) / (std + 1e-9)
            return norm.cdf(z)
    
    def optimize(self, n_iter):
        # Initial random sampling
        X_init = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            size=(self.init_points, len(self.bounds))
        )
        y_init = [self.f(x) for x in X_init]
        self.X.extend(X_init)
        self.y.extend(y_init)
        
        for i in range(n_iter):
            X = np.array(self.X)
            y = np.array(self.y)
            
            # Fit GP model
            gp = GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=5,
                random_state=None
            )
            gp.fit(X, y)
            
            y_best = np.min(y)
            
            # Optimize acquisition function
            x_tries = np.random.uniform(
                self.bounds[:, 0], 
                self.bounds[:, 1],
                size=(100, len(self.bounds))
            )
            ys = self.acquisition_function(x_tries, gp, y_best)
            x_max = x_tries[ys.argmax()]
            
            # Fine-tune the maximum using scipy's minimize
            res = minimize(
                lambda x: -self.acquisition_function(x, gp, y_best),
                x_max,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if not res.success:
                continue
                
            # Sample point
            self.X.append(res.x)
            self.y.append(self.f(res.x))
        
        best_idx = np.argmin(self.y)
        return np.array(self.X[best_idx]), self.y[best_idx]
    


# Optimize simple 2D function
def objective(x):
    return -(x[0]**2 + x[1]**2)

bounds = [(-5, 5), (-5, 5)]
bo = BayesianOptimization(objective, bounds)
x_best, y_best = bo.optimize(n_iter=20)