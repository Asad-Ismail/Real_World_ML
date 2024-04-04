## Interpretable Vs Explainable AI

#### Interpretable AI
Interpretability refers to the extent to which a human can understand the cause of a decision made by an AI model. An interpretable AI model is one whose workings can be comprehended by humans without any additional aid. This means that the model's decision-making process is transparent and can be followed step-by-step. Interpretable models often have simpler structures, such as decision trees, linear regression, or logistic regression, where the relationship between input variables and the prediction is straightforward.

#### Key Characteristics:

**Simplicity**: The model's structure allows for easy understanding of how input features affect the output.
**Transparency**: The internal workings of the model are accessible and clear to the users.
**Direct Understanding**: The user can directly follow how the model's input is transformed into an output.

#### Explainable AI

Explainable AI, on the other hand, involves complex models that are inherently opaque, such as deep neural networks, where interpretability is not directly possible due to the complexity and the large number of parameters involved. XAI seeks to provide explanations for the model's decisions post hoc, using various techniques and tools that can help elucidate how the model arrived at its conclusions. These explanations may come in the form of feature importance scores, visualizations, or surrogate models that approximate the original model in a more interpretable form.

#### Key Characteristics:

**Post Hoc Explanation**: Provides insights into the model’s decisions after the fact, often through additional tools or methods.
**Complex Models**: Applies to models where direct interpretability is not feasible due to their complexity.
**User Trust and Understanding**: Aims to build trust and understanding by explaining the model’s outputs, even if the model's internal workings remain complex and opaque.




## Explainable AI for Stuctured Data/Models

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





