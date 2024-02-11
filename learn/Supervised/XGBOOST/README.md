## XGBOOST

### Gradient Boosting
In a true gradient boosting framework, the algorithm improves upon its predecessors by focusing on the residuals or the gradients of the loss function with respect to the predictions. This approach systematically reduces the loss by moving in the direction opposite to the gradient, hence the name "gradient boosting." The key steps involve calculating the gradient of the loss function for each sample and using these gradients to fit the next learner to better predict the residuals or corrections to the previous learners' predictions.