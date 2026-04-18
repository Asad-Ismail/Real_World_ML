import numpy as np


def standardize_features(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert each feature to zero mean and unit variance.

    Returns the standardized data along with the feature means and standard
    deviations so the same transform can be reused on new data.
    """
    data = np.asarray(data, dtype=float)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized = (data - mean) / (std + 1e-8)
    return standardized, mean, std


if __name__ == "__main__":
    sample = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 12.0, 90.0],
            [3.0, 14.0, 110.0],
            [4.0, 16.0, 95.0],
        ]
    )
    standardized, mean, std = standardize_features(sample)
    print("Feature means:", mean)
    print("Feature std:", std)
    print("Standardized data:")
    print(standardized)
