# Use Cases

This folder contains practical ML examples that sit closer to applied workflows than the `learn/` directory.

## Best Starting Points

- `learning_with_less/`
  - Runnable locally.
  - Covers supervised, self-supervised, and semi-supervised learning with an offline fallback dataset.
- `SparkImageProcessing/`
  - Runnable locally if you have OpenCV and optionally Spark.
  - Good for understanding the difference between a simple local pipeline and a Spark-style batch pipeline.
- `RealTimeDataProcessing/`
  - Runnable if you have Kafka and Spark installed.
  - Shows the shape of a streaming fraud-detection workflow.

## Requires External Services Or Keys

- `Agents/`
  - LLM agent demos.
  - Typically require `MGA_API_KEY` and the relevant Python packages.
- `LLMs/`
  - DSPy and research-generation examples.
  - Typically require `OPENROUTER_API_KEY`, and some scripts also require `TAVILY_API_KEY`.

## Cloud Or Notebook Heavy Examples

- `Sagmeaker_Fraud_Detection_End_To_End/`
  - SageMaker notebook workflow.
- `demand_prediction/`
  - Notebook examples.

## Media-Specific Example

- `sportanalytics/`
  - YOLO-based sports video tracking demo.
  - Requires a local video file and the provided `yolo11n.pt` weights.

## Working Rule

If you want examples that are easiest to study from first principles, start with the folders above that now have:

- clear READMEs
- relative paths instead of machine-specific paths
- import-safe Python scripts
- local fallbacks where possible
