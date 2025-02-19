# Learning with Less Data

In many real-world applications, collecting a large, well-labeled dataset is challenging or even infeasible. We present methods and practices that allow models to learn effectively from limited data. The key approaches include using Pretrained Models, Self-Supervised Learning, and Semi-Supervised Learning. Each of these techniques offers unique advantages depending on your data availability and task requirements.

## 1. Pretrained Models

**Overview:**  
Pretrained models are networks that have been trained on large-scale, diverse datasets (like ImageNet or BERT's training corpus) and can be fine-tuned for a specific task. These models capture a wealth of general features and representations that are transferable to tasks with limited data.

**When to Use:**  
- **Limited Labeled Data:** If your target dataset is small but comes from a **similar domain** as the dataset on which the model was pretrained.
- **Time & Resources Constraints:** Fine-tuning is often significantly faster and less resource-intensive than training from scratch.
- **Robust Baseline:** Pretrained models typically provide strong performance even with minimal fine-tuning.

**Benefits & Considerations:**
- **Pros:**  
  - Fast convergence during training.
  - Excellent generalization on similar tasks.
- **Cons:**  
  - May not fully capture domain-specific nuances if the pretraining domain is too different.

## 2. Self-Supervised Learning

**Overview:**  
Self-supervised learning leverages large quantities of unlabeled data by formulating a pretext task where the data provides its own supervision. For example, predicting missing parts of data, enforcing consistency under data augmentation, or using contrastive methods to learn representations.

**When to Use:**  
- **Abundant Unlabeled Data:** When you have a lot of unlabeled examples, self-supervised learning can help extract meaningful features.
- **Pretext Task Alignment:** When it is possible to define a pretext task that aligns well with the features needed in the downstream task.

**Benefits & Considerations:**
- **Pros:**  
  - Reduces dependency on annotated data.
  - Captures useful representations that generalize well.
- **Cons:**  
  - Designing an effective pretext task can be challenging.
  - Computational expensive, resources are required to train the model on the pretext task.
  - The learned representations might not be perfectly optimized for the final task without further fine-tuning.

## 3. Semi-Supervised Learning

**Overview:**  
Semi-supervised learning combines a small amount of labeled data with a larger pool of unlabeled data during training. The idea is to propagate label information from labeled to unlabeled examples, thereby improving model performance when annotations are scarce.

**When to Use:**  
- **Mixed Data Availability:** When you have a limited set of labeled data but still have large amounts of unlabeled data.
- **Cost Constraints:** When labeling data is expensive or time-consuming, semi-supervised methods can bridge the gap.

**Example Methods:**  
- **Pi Model:** A consistency-based approach that encourages similar predictions for an input and its augmented version.
- Other methods include Mean Teacher, Virtual Adversarial Training, and graph-based methods.

**Benefits & Considerations:**
- **Pros:**  
  - Improves performance over purely supervised learning with small datasets.
  - Efficient use of available unlabeled data.
- **Cons:**  
  - Can be sensitive to the quality and distribution of the unlabeled data.
  - Requires careful balancing between labeled and unlabeled loss components during training.

## Further Reading

For a deeper dive into these concepts, especially semi-supervised techniques, consider reading Lilian Weng's excellent three-part blog series on the subject:  
[Part 1: Semi-Supervised Learning](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/)

## Summary

- **Pretrained Models:**  
  Ideal when you have limited labeled data and a well-established pretrained architecture exists in your domain. Start here if rapid prototyping and leveraging established features are your goals.

- **Self-Supervised Learning:**  
  Best suited when you have abundant unlabeled data. It’s an effective way to learn robust representations before fine-tuning on your specific task.

- **Semi-Supervised Learning:**  
  Use when you face a moderate amount of labeled data combined with a large amount of unlabeled data. It’s particularly useful when labeling is expensive and there exists a reliable method to propagate labels.


## Implemntation

Recent studies ([Semi vs Unsupervised](https://arxiv.org/pdf/2307.08919)) have shown that MixMatch performs remarkably well. [MixMatch](https://arxiv.org/pdf/1905.02249) that incorporates mixup with semi-supervised techniques. Combining selfsupervied pretraining and semi-supervised finetuning also is shown to perform quite well as show in [SelfMatch](https://arxiv.org/pdf/2101.06480) and [CoMatch](https://arxiv.org/pdf/2011.11183). So here we will implement Self supervised pretraining and Semi-supervised finetuning, it is bit different from the original paper of SelfMatch where they use FixMatch for semi-supervised finetuning.


## Results: Age Prediction from Facial Images

### Dataset Statistics

| Split | Number of Images | 
|-------|-----------------|
| Total Dataset | 8,469 | 
| Training (Labeled) | 423 |
| Validation | 254 |
| Unlabeled | 7,792(For semisupervised and self supervised) |

### Performance Comparison (RMSE)

| Training Approach | Validation RMSE | Improvement |
|------------------|-----------------|-------------|
| Training from Scratch | 272.00 | Baseline |
| ImageNet Pretrained (Supervised) | 94.57 | -65.23% |
| ImageNet Pretrained + Semi-supervised | 79.56 | -70.75% |
| Self-supervised + Semi-supervised | 119.00 | -56.25% |

### Key Findings
 
1. **Pretrained Model is Crucial**: 
  - Despite the domain difference between ImageNet and facial images, using pretrained weights dramatically improved performance (65% reduction in error)

2. **Semi-supervised Advantage**: 
  - Adding semi-supervised learning to ImageNet pretraining further reduced error by ~16%
  - Best performance achieved by combining ImageNet weights with semi-supervised learning

2. **Limited Self-supervised Success**: 
  - Self-supervised + semi-supervised approach, while better than training from scratch, couldn't outperform ImageNet pretraining
  - This is likely due to:
    - Limited dataset size (8K images vs ImageNet's 1M+ images)
    - Insufficient data for effective self-supervised representation learning

3. **Semi-supervised Advantage**: 
  - Adding semi-supervised learning to ImageNet pretraining further reduced error by ~16%
  - Best performance achieved by combining ImageNet weights with semi-supervised learning

### Recommendations

1. **Start with Pretrained Models**:
  - Even for domain-specific tasks, ImageNet weights provide a strong foundation
  - Particularly valuable when working with limited labeled data

2. **Consider Dataset Size**:
  - Self-supervised learning might require larger datasets to be effective
  - If working with small datasets (<10K images), ImageNet pretraining might be more beneficial

3. **Add Semi-supervised Learning**:
  - If you have unlabeled data, semi-supervised learning can provide additional gains
  - Works well in combination with pretrained weights










