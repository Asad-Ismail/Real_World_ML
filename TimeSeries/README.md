
# Time Series Analysis

### Time Series Decomposition

Time series decomposition involves breaking down a time series into several components, each representing underlying patterns. The primary components are:

### 1. Level
- **Definition**: The level of a time series represents the baseline value around which the series fluctuates. It signifies the average value in the series over time.
- **Significance**: Understanding the level is crucial for setting a starting point for the series and serves as a benchmark for measuring deviations of other components.

### 2. Trend
- **Definition**: The trend shows the long-term progression or direction of the time series data, indicating a consistent increase or decrease over time.
- **Significance**: Identifying the trend is important for understanding the general direction of the time series, which aids in making long-term forecasts and in the analysis of the overall progression.

### 3. Seasonality
- **Definition**: Seasonality reflects regular and predictable patterns that occur within a fixed period, such as daily, monthly, or quarterly fluctuations.
- **Significance**: Recognizing seasonality is crucial for anticipating regular patterns and adjusting business strategies accordingly. It allows for more accurate analysis and forecasting by accounting for regular fluctuations.

### 4. Noise (or Irregular Component)
- **Definition**: Noise is the random or irregular fluctuations in the time series that are not explained by the level, trend, or seasonal components. It represents the unexplained variance in the series.
- **Significance**: Understanding noise helps in refining models. High noise levels might indicate that a model isn't capturing some aspects of the data, and reducing noise can lead to better predictions and understanding of the underlying structure.

Decomposing a time series provides a clearer understanding of its underlying patterns, which contributes to more accurate and meaningful models for forecasting and analysis.



## Understanding Stationarity in Time Series

Stationarity is a fundamental concept in time series analysis that refers to a time series whose statistical properties do not change over time. This includes aspects like the mean, variance, and autocorrelation. For a time series to be stationary, it must meet the following criteria:

- **Constant Mean**: The average value of the series should not vary over time.
- **Constant Variance**: The variance of the series should remain consistent over time, meaning the series does not exhibit periods of increased or decreased volatility.
- **Constant Autocorrelation Structure**: How the data points are correlated with each other should not change.

### Why is Stationarity Important?

Stationarity is a crucial assumption in many time series models because models built on stationary data are more reliable and have more valid forecasts. Non-stationary data, due to their changing statistical properties, can lead to misleading models and forecasts.

### Detecting Non-Stationarity:

1. **Visual Inspection**: 
   - Plotting the time series data can provide an initial check. Look for changing mean or variance over time.
   - Plotting rolling statistics: Observing the rolling mean and rolling standard deviation over time can help identify non-stationarity.

2. **Statistical Tests**: 
   - **Dickey-Fuller Test**: A common statistical test used to determine the presence of a unit root in the series, which indicates non-stationarity.
     - Null Hypothesis (\(H_0\)): The series has a unit root (is non-stationary).
     - Alternative Hypothesis (\(H_1\)): The series has no unit root (is stationary).
     - If the test statistic is less than the critical value, we reject the null hypothesis and infer that the series is stationary.
   - **KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)**: Another test where the null hypothesis is that the series is stationary.
     - A significant p-value indicates the series is non-stationary.

3. **Autocorrelation and Partial Autocorrelation Plots**:
   - Observing the AC and PAC plots can give insights into the stationarity of the series. For a stationary series, the autocorrelations typically quickly decay to zero.

### Handling Non-Stationarity:

- **Differencing**: Subtracting the current value from the previous removes trends and cycles, often leading to stationarity.
- **Transformation**: Applying transformations like logarithms or square roots can stabilize the variance.
- **Detrending**: Removing the trend component from the series.

Understanding and ensuring stationarity in time series data is vital before proceeding with further analysis or modeling, as it impacts the effectiveness and reliability of statistical inferences and predictive models.








## References & Acknoledgment

https://github.com/jiwidi/time-series-forecasting-with-python/tree/master

https://github.com/PacktPublishing/Modern-Time-Series-Forecasting-with-Python/tree/main