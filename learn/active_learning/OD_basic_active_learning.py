from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionPrediction:
    confidences: np.ndarray


class ObjectDetectionModel:
    def __init__(self, seed=42, min_detections=1, max_detections=5):
        self.seed = seed
        self.min_detections = min_detections
        self.max_detections = max_detections

    def predict(self, image_id):
        rng = np.random.default_rng(self.seed + image_id)
        num_detections = int(rng.integers(self.min_detections, self.max_detections + 1))
        logits = rng.normal(loc=0.0, scale=1.0, size=num_detections)
        confidences = 1.0 / (1.0 + np.exp(-logits))
        return DetectionPrediction(confidences=confidences)


def object_score_uncertainty(prediction, threshold=0.5):
    distance_from_boundary = np.abs(prediction.confidences - threshold)
    mean_distance = np.mean(distance_from_boundary)
    uncertainty = 1.0 - 2.0 * mean_distance
    return float(np.clip(uncertainty, 0.0, 1.0))


def rank_unlabeled_examples(unlabeled_dataset, model, n_queries=5):
    scored_examples = []

    for image_id in unlabeled_dataset:
        prediction = model.predict(image_id)
        uncertainty = object_score_uncertainty(prediction)
        scored_examples.append(
            {
                "image_id": image_id,
                "num_detections": int(len(prediction.confidences)),
                "mean_confidence": float(np.mean(prediction.confidences)),
                "uncertainty": uncertainty,
            }
        )

    scored_examples.sort(key=lambda item: item["uncertainty"], reverse=True)
    return scored_examples[:n_queries], scored_examples


if __name__ == "__main__":
    labeled_dataset = list(range(10))
    unlabeled_dataset = list(range(10, 30))

    model = ObjectDetectionModel(seed=42)
    selected_examples, all_scores = rank_unlabeled_examples(unlabeled_dataset, model, n_queries=5)

    print(f"Labeled pool size: {len(labeled_dataset)}")
    print(f"Unlabeled pool size: {len(unlabeled_dataset)}")
    print("Most informative images to label next:")
    for example in selected_examples:
        print(
            f"image_id={example['image_id']}, "
            f"num_detections={example['num_detections']}, "
            f"mean_confidence={example['mean_confidence']:.3f}, "
            f"uncertainty={example['uncertainty']:.3f}"
        )

    print("Active learning completed!")
