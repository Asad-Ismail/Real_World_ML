## Feature Engineerig:

Unsupervised Methods:
    1. Removing features with low variance

Supervised Methods:
    1. Slect K Best / Select Percentile
        1. Classification

            #### Pearson correlation coefficent range between (-1,-1)

            import numpy as np
            # Sample data
            heights = np.array([150, 160, 170, 180])
            weights = np.array([50, 60, 70, 80])

            # Calculate Pearson's correlation coefficient
            r = np.corrcoef(heights, weights)[0, 1]

            print(f"Pearson's R: {r:.2f}")

            #### ANOVA test
        2. Regression
            #### Linear Regression


    2. Feature selection using SelectFromModel
        1.  L1-based feature selection
             Lasso for regression, aLogisticRegression and LinearSVC for classification
        2. Tree-based feature selection








