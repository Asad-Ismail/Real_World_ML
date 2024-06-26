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

### Permutation Importance

Permutation Importance is a straightforward approach to gauge the significance of each feature in your dataset relative to the model's performance.

#### Steps:

1. **Initialize with a Trained Model:** Begin with your model that has already been trained on your dataset.
2. **Evaluate Feature Importance:**
   - For each feature column in the dataset, shuffle its values. This disrupts the relationship between the feature and the outcome.
   - With the shuffled dataset, make predictions and assess the impact on the model's accuracy or performance metric. The degradation in performance signifies the feature's importance.
   - Restore the data to its original order before proceeding to the next feature.
3. **Iterate:** Repeat the process for each feature to compile a comprehensive importance ranking.

### Partial Dependence Plots (PDP)

Partial Dependence Plots (PDP) provide insights into the relationship between features and the prediction outcome, across a range of values.

#### Concept:

- PDPs use the trained model to predict outcomes, systematically altering one feature value at a time, while keeping others constant.
- By observing how predictions change with varying levels of a single feature, we can infer the relationship between that feature and the outcome.
- PDPs are akin to observing coefficients in linear models but are applicable to more complex scenarios, offering a graphical representation of the feature's effect on predictions.

### SHAP Values (SHapley Additive exPlanations)

SHAP Values offer a comprehensive breakdown of each feature's contribution to a prediction, based on cooperative game theory.

#### Concept:

- SHAP values articulate the contribution of each feature to the difference between the actual prediction and a baseline prediction.
- They ensure an equitable distribution of contribution across all features, encapsulated by the formula:
  ```
  sum(SHAP values for all features) = prediction for instance - prediction for baseline
  ```
**Example**:

If we use the median house price from our training dataset as the baseline, our baseline prediction is the median value. This means that if we had no information about a house's features (size, location, number of bedrooms, etc.), our best guess would be the median price.

When we calculate SHAP values or any other feature attribution method, we are essentially asking: "How does knowing the size of the house, its location, etc., change our prediction from this baseline (median) prediction?"



## Explainability for Images


### Occlusion Based

Occlusion based method moves mask of certain size across image and checks the difference /effect on the output. The part of image which effects most the prediction is the most relevant for that prediction

- **Advantages:** Easy to understand and implement
- **Limitations:** Computational expensive/infessible to implement for high resolution images


### 1. Saliency Maps

Saliency maps highlight the parts of an input image that are most influential in determining the model's decision. These maps can be generated by computing the gradient of the output with respect to the input image, indicating how much each pixel contributes to the final decision. This helps in visualizing which parts of the image are most important for classification or detection tasks.

- **Advantages:** Provides a direct, visual way to understand which pixels influence the model's decision.
- **Limitations:** Might be noisy and not always perfectly interpretable for complex images.

### 2. Class Activation Mapping (CAM) and its Variants (Grad-CAM, Grad-CAM++)

CAM techniques generate heatmaps that indicate the regions of the image most relevant to a particular class label. These methods rely on the spatial information preserved in the feature maps of CNNs to identify the image regions most important for discriminating between different classes.

- **Grad-CAM:** Uses the gradients of any target concept (output neuron) flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.
- **Advantages:** Easy to implement and apply to a wide range of CNN models without modification.
- **Limitations:** Limited resolution of the heatmap, which may not accurately represent all the nuances of what the model is paying attention to.

### 3. Layer-wise Relevance Propagation (LRP)

LRP is a technique that attributes the prediction of a deep network to its input components, essentially reversing the forward pass of a network. It propagates the prediction backward through the network layers to assign a relevance score to each input pixel, showing how much each pixel contributed to the final decision.

- **Advantages:** Provides pixel-wise decomposition of the model's decision, offering detailed insight into the model's reasoning.
- **Limitations:** Implementation can be complex and model-specific.

### 4. Counterfactual Explanations

Counterfactual explanations provide insight into model decisions by answering hypothetical questions: "What would need to change for a different decision to be made?" For images, this could involve modifying parts of an image to see how these changes affect the model's prediction, essentially identifying the changes needed to alter the decision.

- **Advantages:** Intuitive and easy to understand, as it directly relates to changes that can flip the model's decision.
- **Limitations:** Generating realistic counterfactuals for images can be challenging and computationally intensive.

### 5. Feature Visualization

Feature visualization involves enhancing the input image to maximize the activation of specific neurons in the model. This technique helps in understanding what features a particular layer of the network is looking for in the inputs it processes.

- **Advantages:** Offers insights into the learned features at various layers of the network.
- **Limitations:** The highly activated features might not always correspond to human-intuitive concepts.




  






