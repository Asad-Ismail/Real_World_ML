# Linear Discriminant Analysis (LDA)

**Linear Discriminant Analysis (LDA)** is a statistical method utilized in machine learning and pattern recognition for dimensionality reduction and classification purposes. It is a supervised learning technique, requiring labeled training data to operate effectively.

## Primary Goal

The primary objective of LDA is to identify a linear combination of features that optimally separates two or more classes. This technique projects the data onto a lower-dimensional subspace where the separation between different classes is maximized, and the scatter within each class is minimized.

## Key Steps in LDA

1. **Compute the Class Means**: Calculate the mean vector for each class in the dataset.

2. **Compute the Within-Class Scatter Matrix**: Calculate the scatter matrix for each class and sum them to obtain the overall within-class scatter matrix.

3. **Compute the Between-Class Scatter Matrix**: Calculate the difference between the class mean vectors and the overall mean, weighted by the number of samples in each class. The outer product of these weighted differences is then computed and summed to derive the overall between-class scatter matrix.

4. **Compute the Eigenvectors and Eigenvalues**: Solve the generalized eigenvalue problem for the within-class scatter matrix versus the between-class scatter matrix to obtain eigenvectors and eigenvalues.

5. **Select the Most Discriminant Features**: Select the top *k* eigenvectors with the largest eigenvalues, where *k* is the desired number of dimensions for the reduced data. These eigenvectors are the linear discriminants that form the new axes for data projection.

6. **Project the Data**: Transform the original dataset by projecting it onto the new, lower-dimensional subspace defined by the selected linear discriminants.

## Application and Assumptions

After projection, classification tasks can be performed using classifiers like logistic regression or support vector machines. LDA is particularly beneficial when the feature set is large relative to the sample size and when data approximately follows a Gaussian distribution with equal covariance matrices across each class. It finds extensive application in facial recognition, medical diagnosis, and text classification.

### Assumptions
- Gaussian distributed data for each class.
- Equal covariance matrices for each class.
- Linearity in the separation of classes.

LDA's effectiveness is subject to these assumptions, which should be considered when choosing LDA for a specific problem.
