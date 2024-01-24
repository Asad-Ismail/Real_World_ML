### General Interview Tips for ML/Data Scientist Interview

## Machine Learning Pipeline for Fraud Detection with Tabular Data

### Step 1: Business Requirements
- Real-time vs Batch Processing
- Defining the problem statement
- Identifying key stakeholders and their needs

### Step 2: Data Sources/Labeling for Fraud Detection

#### 1. User Data:
- **Account Creation Date**: To identify newer accounts which might be more prone to fraudulent activities.
- **Login History**: Analysis of frequent logins or logins from unusual locations.
- **Payment History**: Past payment behaviors, including chargebacks or disputes.
- **User Behavior Patterns**: Typical transaction amounts, times, and frequencies.

#### 2. Device Data:
- **Device ID/Fingerprint**: Unique identifier for devices used in transactions.
- **Browser Information**: Type and version of the browser used.
- **Mobile Carrier (for mobile transactions)**: For geolocation and fraud pattern identification.

#### 3. Transaction Data:
- **Merchant Category**: Type of merchant where the transaction occurred.
- **Shipping Address (for e-commerce)**: Comparison with billing address or user's known addresses.
- **Payment Method Used**: Different payment methods and their associated risk levels.

#### 4. Network Data:
- **IP Address Analysis**: Geolocation, consistency with previous transactions.
- **VPN or Proxy Detection**: Usage of VPN or proxy as a potential fraud indicator.

#### 5. Labeling Strategies:
- **Supervised Labels**: Distinguishing between fraudulent and non-fraudulent transactions based on historical data.
- **Unsupervised Techniques**: For detecting new types of fraud without pre-labeled data.
- **Feedback Loops**: Incorporating user-reported fraud to update labels continually.

### Step 3: Data Exploration and Preprocessing
- Exploratory Data Analysis (EDA): Missing values, univariate analysis, feature distributions
- Data Cleaning: Handling missing values, outliers
- Data Preprocessing: Normalization, one-hot encoding, grouping, feature scaling
- Class Imbalance hangling (Oversampling, undersampling, SMOTE)
 

### Step 4: Feature Engineering

#### 1. Domain-Specific Features:

**Transaction Behavior** : Features like average transaction amount, frequency of transactions, and deviation from the user's normal transaction patterns.


**User Account Features** : Age of the account, frequency of password changes, number of failed login attempts.


**Geographical Features** : Mismatch between the user's known location and the transaction location.

#### 2. Time-Based Features:

**Temporal Patterns**: Time of the transaction can be crucial. For instance, transactions in the middle of the night might be more suspicious.

#### 3. Aggregation Features:

**Statistical Aggregates** : Mean, median, and standard deviation of user transactions over a certain period.

**Rolling Window Statistics** : Calculating aggregates over rolling windows (e.g., 7-day rolling average of transaction amounts).

### 4. Feature Selection:

**PCA**

**Lasso L1 Regression**

**Ridge L2 Regression**




### Step 5: Initial Evaluation Metrics/KPIs
- Defining Key Performance Indicators (KPIs)
- Initial Metrics: Precision, Recall, F1-Score, ROC-AUC for classification tasks

### Step 6: Model Selection and Training
- Choosing appropriate models (e.g., Decision Trees, Random Forest, Gradient Boosting, Neural Networks)
- Training models on the preprocessed dataset
- HP tuning

### Step 7: Model Evaluation
- Offline Evaluation: Cross-validation, confusion matrix, ROC curves

#### Step 8. Model Interpretability:

- Permutation Feature Importance: Assess the increase in the model's prediction error after permuting each feature. More error implies higher importance.
- Partial Dependence Plots (PDPs): Show the effect of a single feature on the predicted outcome of a model.
-  Use Interpretable models
-  LIME 
-  SHAP

### Step 8: Online Evaluation (if applicable)
- A/B Testing
- Real-time model performance monitoring

### Step 9: Deployment
- Model deployment strategies (e.g., cloud, on-premises)
- Integration with existing systems
- Deployment tools and technologies (e.g., Docker, Kubernetes)

### Step 10: Monitoring and Maintenance
- Continuous monitoring of model performance
- Logging and alert systems
- Periodic model retraining and updating

### Step 11: Continual Learning
- Strategies for updating the model with new data
- Techniques for addressing concept drift

### Step 12: Ethics and Compliance
- Ensuring data privacy and security
- Compliance with legal and ethical standards
- Bias and fairness in model predictions
- Permutation Test
- Invariance Test
- Directional Test
- Differential Privavy
