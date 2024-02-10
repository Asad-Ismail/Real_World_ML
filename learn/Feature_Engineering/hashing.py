from sklearn.feature_extraction import FeatureHasher

# Example dataset with categorical features
data = [
    {'feature1': 'dog', 'feature2': 'blue'},
    {'feature1': 'cat', 'feature2': 'green'},
    {'feature1': 'bird', 'feature2': 'blue'},
    {'feature1': 'fish', 'feature2': 'green'},
]

# Initialize the FeatureHasher
# n_features specifies the size of the hashing space.
# input_type='dict' indicates that each input is a dictionary where keys are feature names and values are feature values.
hasher = FeatureHasher(n_features=2, input_type='dict')

# Transform the dataset using the hashing trick
transformed_data = hasher.transform(data)

# Convert the transformed data to an array for easy viewing
transformed_array = transformed_data.toarray()

print(transformed_array)
