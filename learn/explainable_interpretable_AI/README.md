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

Permutation Importance is a straightforward approach to gauge the significance of each feature in your dataset relative to the model's performance.

### Steps:

1. **Initialize with a Trained Model:** Begin with your model that has already been trained on your dataset.
2. **Evaluate Feature Importance:**
   - For each feature column in the dataset, shuffle its values. This disrupts the relationship between the feature and the outcome.
   - With the shuffled dataset, make predictions and assess the impact on the model's accuracy or performance metric. The degradation in performance signifies the feature's importance.
   - Restore the data to its original order before proceeding to the next feature.
3. **Iterate:** Repeat the process for each feature to compile a comprehensive importance ranking.

## Partial Dependence Plots (PDP)

Partial Dependence Plots (PDP) provide insights into the relationship between features and the prediction outcome, across a range of values.

### Overview:

- PDPs use the trained model to predict outcomes, systematically altering one feature value at a time, while keeping others constant.
- By observing how predictions change with varying levels of a single feature, we can infer the relationship between that feature and the outcome.
- PDPs are akin to observing coefficients in linear models but are applicable to more complex scenarios, offering a graphical representation of the feature's effect on predictions.

## SHAP Values (SHapley Additive exPlanations)

SHAP Values offer a comprehensive breakdown of each feature's contribution to a prediction, based on cooperative game theory.

### Concept:

- SHAP values articulate the contribution of each feature to the difference between the actual prediction and a baseline prediction.
- They ensure an equitable distribution of contribution across all features, encapsulated by the formula:
  ```
  sum(SHAP values for all features) = prediction for instance - prediction for baseline
  ```
**Example**:

If we use the median house price from our training dataset as the baseline, our baseline prediction is the median value. This means that if we had no information about a house's features (size, location, number of bedrooms, etc.), our best guess would be the median price.

When we calculate SHAP values or any other feature attribution method, we are essentially asking: "How does knowing the size of the house, its location, etc., change our prediction from this baseline (median) prediction?"



  






