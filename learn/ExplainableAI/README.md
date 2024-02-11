
## Permutation Importance

1. Get a trained model.
2. Shuffle the values in a single column, make predictions using the resulting dataset. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.
3. Return the data to the original order (undoing the shuffle from step 2). Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column.


## Partial Dependencies plot (PDP)
We  use the fitted model to predict outcome. But we repeatedly alter the value for one variable at a time to make a series of predictions. We vary this variable from low to high and measure the differnce in predictio for all data points. Can be interpreted as coefficents in liner/logistic regression but it can be used for much complex models


## SHAP Values (SHapley Additive exPlanations)
How much was a prediction driven by the fact that the feature has that value, instead of some baseline value.
SHAP values do this in a way that guarantees a nice property. Specifically, you decompose a prediction with the following equation:

sum(SHAP values for all features) = pred_for_team - pred_for_baseline_values


