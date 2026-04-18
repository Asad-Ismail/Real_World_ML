import numpy as np


def softmax(logits, axis=0):
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


class SegmentationModel:
    def __init__(self, seed=42, n_classes=3, height=16, width=16):
        self.seed = seed
        self.n_classes = n_classes
        self.height = height
        self.width = width

    def predict_logits(self, image_id):
        rng = np.random.default_rng(self.seed + image_id)
        return rng.normal(size=(self.n_classes, self.height, self.width))


def segmentation_entropy_uncertainty(logits):
    probabilities = softmax(logits, axis=0)
    pixel_entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=0)
    return float(np.mean(pixel_entropy))


def rank_unlabeled_examples(unlabeled_dataset, model, n_queries=5):
    scored_examples = []

    for image_id in unlabeled_dataset:
        logits = model.predict_logits(image_id)
        uncertainty = segmentation_entropy_uncertainty(logits)
        scored_examples.append({"image_id": image_id, "uncertainty": uncertainty})

    scored_examples.sort(key=lambda item: item["uncertainty"], reverse=True)
    return scored_examples[:n_queries], scored_examples


if __name__ == "__main__":
    unlabeled_dataset = list(range(20))
    model = SegmentationModel(seed=42)
    selected_examples, _ = rank_unlabeled_examples(unlabeled_dataset, model, n_queries=5)

    print("Images to label next based on entropy:")
    for example in selected_examples:
        print(f"image_id={example['image_id']}, uncertainty={example['uncertainty']:.4f}")

    print("Active learning completed!")
