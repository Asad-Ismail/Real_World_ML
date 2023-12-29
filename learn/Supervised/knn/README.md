# K-Nearest Neighbor (KNN)

K-Nearest Neighbor (KNN) is a machine learning algorithm that is often used for classification and regression problems. KNN is a non-parametric method, which means that it does not make any assumptions about the distribution of the data. Instead, it relies on the nearest neighbors to make predictions.

In KNN, the number "k" represents the number of nearest neighbors that are considered when making a prediction. When new data is inputted into the model, KNN will calculate the distance between that point and all the other points in the dataset. It will then choose the "k" closest points and use their labels (if it's a classification problem) or their values (if it's a regression problem) to predict the label or value of the new data.

KNN is a simple and easy-to-implement algorithm that can be used for both classification and regression problems. It is particularly useful when there is little or no prior knowledge about the data, as it does not require any assumptions to be made about the distribution of the data. However, KNN can be sensitive to the choice of "k" and the distance metric used, and can be computationally expensive for large datasets.

We have implmented KNN for classification for implementation