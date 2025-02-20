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


## Example Implementation and General Recommendations

Lets demostrate it using example of age prediction from facial images an image regression task. Our experiments were conducted on a dataset size (~8K images) is also similar to many real-world scenarios, especially in specialized domains like:

- Medical imaging (tumor detection)
- Agricultural 
- Industrial defect detection

These domains often face similar challenges:
- Limited labeled data due to expensive annotation
- Domain expertise required for labeling

### Dataset Statistics

| Split | Number of Images | 
|-------|-----------------|
| Total Dataset | 8,469 | 
| Training (Labeled) | 423 |
| Validation | 254 |
| Unlabeled | 7,792 (For semisupervised and self supervised) |

### Performance Comparison

Below are results of our experiment, we are not trying to get SOTA performance but rather compare performance imporvement using different methods given all other parameters like batch size and learning rate stays the same. 

*Note: Lower MSE indicates better performance*. 

| Training Approach | Validation MSE | Improvement |
|------------------|-----------------|-------------|
| Training from Scratch | 272.00 | Baseline |
| ImageNet Pretrained (Supervised) | 94.57 | -65.23% |
| ImageNet Pretrained + Semi-supervised | 79.56 | -70.75% |
| Self-supervised + Semi-supervised | 119.00 | -56.25% |

### Key Findings
 
1. **Pretrained Models are good starting point**: 
  - Despite the domain difference between ImageNet and facial images, using pretrained weights dramatically improved performance (65% reduction in error)

2. **Semi-supervised Advantage**: 
  - Adding semi-supervised learning to ImageNet pretraining further reduced error by ~16% (over pretrained model training)
  - Best performance achieved by combining ImageNet weights with semi-supervised learning

3. **Limited Self-supervised Success**: 
  - Self-supervised + semi-supervised approach, while better than training from scratch, couldn't outperform ImageNet pretraining
  - This is likely due to:
    - Limited dataset size (8K images vs ImageNet's 1M+ images)
    - Insufficient data for effective self-supervised representation learning

### Recommendations

Below are some recommendations of training DNNs specifically true on image related tasks(classificaiton, semgmentation, OD etc)

1. **Start with Pretrained Models**:
  - Even for domain-specific tasktion
  - Particularly valuable when working with limited labeled data
    
2. **Leverage Data Augmentation**:
  - Add task-specific augmentations they are consistently shown to improve model performance some image sepecific augmentation are MixUp/CutMix, CutOut, Color jittering etc
  - Some augmentations might hurt performance if not aligned with task

3. **Tune Hyperparameters** (Not focus of this article):
  - Often overlooked but time and time again shown to 

4. **Add Semi-supervised Learning**:
  - If you have unlabeled data, semi-supervised learning can provide additional gains
  - Works well in combination with pretrained weights

5. **Self Supervised learning may or maynot be helpful**:
  - Self-supervised learning might require larger datasets to be effective
  - If working with small datasets for example for images < 10K , self supervised might not work good enough for it and ImageNet pretraining might be more beneficial



## Further Reading

For a deeper dive into these concepts, especially semi-supervised techniques, consider reading Lilian Weng's excellent three-part blog series on the subject:  

[Part 1: Semi-Supervised Learning](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/)

[Part 2: Active Learning](https://lilianweng.github.io/posts/2022-02-20-active-learning/)

[Part 3: Data Augmentations](https://lilianweng.github.io/posts/2022-04-15-data-gen/)


@article{weng2022datagen,
  title   = "Learning with not Enough Data Part 3: Data Generation",
  author  = "Weng, Lilian",
  journal = "Lil'Log",
  year    = "2022",
  month   = "Apr",
  url     = "https://lilianweng.github.io/posts/2022-04-15-data-gen/"
}









