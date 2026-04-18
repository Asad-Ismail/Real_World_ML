import numpy as np


def softmax(logits, axis=0):
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


class SegmentationModelWithDropout:
    def __init__(self, seed=42, n_classes=3, height=16, width=16, dropout_rate=0.3):
        self.seed = seed
        self.n_classes = n_classes
        self.height = height
        self.width = width
        self.dropout_rate = dropout_rate

    def forward(self, image_id, rng):
        logits = rng.normal(size=(self.n_classes, self.height, self.width))
        keep_probability = 1.0 - self.dropout_rate
        dropout_mask = rng.binomial(1, keep_probability, size=logits.shape) / max(keep_probability, 1e-8)
        return logits * dropout_mask


def mc_dropout_uncertainty(model, image_id, mc_samples=10):
    rng = np.random.default_rng(model.seed + image_id)
    predictions = []

    for _ in range(mc_samples):
        logits = model.forward(image_id, rng)
        predictions.append(softmax(logits, axis=0))

    stacked_predictions = np.stack(predictions, axis=0)
    return float(np.mean(np.var(stacked_predictions, axis=0)))


def rank_unlabeled_examples(unlabeled_dataset, model, n_queries=5, mc_samples=10):
    scored_examples = []

    for image_id in unlabeled_dataset:
        uncertainty = mc_dropout_uncertainty(model, image_id, mc_samples=mc_samples)
        scored_examples.append({"image_id": image_id, "uncertainty": uncertainty})

    scored_examples.sort(key=lambda item: item["uncertainty"], reverse=True)
    return scored_examples[:n_queries], scored_examples


if __name__ == "__main__":
    unlabeled_dataset = list(range(20))
    model = SegmentationModelWithDropout(seed=42, dropout_rate=0.3)
    selected_examples, _ = rank_unlabeled_examples(unlabeled_dataset, model, n_queries=5, mc_samples=10)

    print("Images to label next based on Monte Carlo dropout:")
    for example in selected_examples:
        print(f"image_id={example['image_id']}, uncertainty={example['uncertainty']:.6f}")

    print("Active learning completed!")
