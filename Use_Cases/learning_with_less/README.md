# Learning with Less Data

In many real-world applications, collecting a large, well-labeled dataset is challenging or even infeasible. We present methods and practices that allow models to learn effectively from limited data. The key approaches include using Pretrained Models, Self-Supervised Learning, and Semi-Supervised Learning. Each of these techniques offers unique advantages depending on your data availability and task requirements.


## 1. Transfer Learning

**Overview:**  
Trnsfer learning from pretrained models are networks that have been trained on large-scale, diverse datasets (like ImageNet or BERT's training corpus) and can be fine-tuned for a specific task. These models capture a wealth of general features and representations that are transferable to tasks with limited data.

**When to Use:**  
- **Limited Labeled Data:** If your target dataset is small but comes from a similar domain* as the dataset on which the model was pretrained.
- **Time & Resources Constraints:** Fine-tuning is often significantly faster and less resource-intensive than training from scratch.
- **Robust Baseline:** Pretrained models typically provide strong performance even with minimal fine-tuning.

**Benefits & Considerations:**
- **Pros:**  
  - Fast convergence during training.
  - Excellent generalization on similar tasks.
  - Also imporves performance 
- **Cons:**  
  - May not fully capture domain-specific nuances if the pretraining domain is too different.

** Even for significantly different domains like medical imaging, starting with pretrained models (e.g., ImageNet weights) typically outperforms random initialization and accelerates convergence. This makes transfer learning valuable even across substantial domain gaps, providing a strong foundation regardless of target task specificity.


## 2. Self-Supervised Learning

**Overview:**  
Self-supervised learning leverages large quantities of unlabeled data by formulating a pretext task where the data provides its own supervision. For example, predicting missing parts of data, enforcing consistency under data augmentation, or using contrastive methods to learn representations.

**When to Use:**  

- **Abundant Unlabeled Data:** When you have a lot of unlabeled examples, self-supervised learning can help extract meaningful features.
- **Pretext Task Alignment:** When it is possible to define a pretext task that aligns well with the features needed in the downstream task.

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

- **Pros:**  
  - Improves performance over purely supervised learning with small datasets.
  - Efficient use of available unlabeled data.
- **Cons:**  
  - It is sensitive to the quality and distribution of the unlabeled data.
  - Requires careful balancing between the labeled and unlabeled loss terms. Often, the unlabeled loss coefficient is linearly ramped up during training.


## 4. Data Augmentation

**Overview:**  
Data augmentation expands the training dataset by applying controlled transformations to existing samples. These transformations should preserve the semantic meaning of the data while introducing meaningful variations that help the model learn robust features.

**When to Use:**  
- **Limited Dataset Size:** When you have a small labelled dataset most of the real world applications so it is almost always a good idea to use data augmentation.
- **Known Invariances:** When you understand what transformations your model should be invariant to (e.g., rotation, lighting changes).
- **Domain-Specific Requirements:** When you can incorporate domain knowledge into augmentation strategies.

**Key Techniques:**
1. **Basic Transformations:**
   - Geometric: Flipping, rotation, scaling, cropping
   - Color: Brightness, contrast, saturation adjustment
   - Noise: Gaussian noise, blur, random erasing

2. **Advanced Methods:**
   - Mixing: MixUp, CutMix (combining multiple images)
   - Learned: AutoAugment, RandAugment (automated policies)
   - Consistency: AugMix (multiple augmented versions)


**Benefits & Considerations:**
- **Pros:**  
  - Increases effective dataset size without additional data collection.
  - Improves model robustness and generalization.
  - Can encode domain-specific invariances.
- **Cons:**  
  - Requires careful selection of augmentation strategies to avoid introducing harmful transformations.

### 5. Hyperparameter Tuning

**Overview:**  
Hyperparameters are parameters that are not learned from data during training, but instead are set by the user before training begins. Hyperparameter tuning is the process of systematically searching for optimal model configuration parameters. 

**When to Use:**  
 - **Not highly Resource Constraints:**
 Almost alwys a good idea to tune hyperparameters for better performance given you have enough computational resources.

**Benefits & Considerations:**
- **Pros:**  
  - Can significantly improve model performance without additional data.
  - Helps identify optimal model complexity for limited data scenarios.
- **Cons:**  
  - Can be computationally expensive.
  - Risk of overfitting to validation set if not done carefully.


## 6. Active Learning

**Overview:**  
Active learning is an iterative process where the model identifies the most informative unlabeled examples for human annotation. This approach optimizes the data labeling effort by focusing on samples that would most improve model performance.

**When to Use:**  
- **Limited Annotation Budget:** When you have constraints on how many samples can be labeled which is a common scenario in real world applications.
- **Large Unlabeled Pool:** When you have access to many unlabeled examples but can only label a subset.
- **Expensive Labeling:** When annotation requires significant time or expertise which is also mostly the case.

**Benefits & Considerations:**
- **Pros:**  
  - Reduces annotation costs while maximizing learning efficiency.
  - Focuses human effort on the most valuable examples.
  - Can achieve better performance with fewer labeled examples.
- **Cons:**  
  - Requires careful design of sample selection strategy.
  - May introduce bias if selection criteria aren't well-designed.
  - Needs interactive labeling infrastructure.



## Example Implementation and General Recommendations

Let's demonstrate this using example of age prediction from facial images an image regression task. Our experiments were conducted on a dataset size (~8K images) is also similar to many real-world scenarios, especially in specialized domains like:

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

5. **Self Supervised learning may or maynot be helpful (if input dataset is limited))**:
  - Self-supervised learning might require larger datasets to be effective
  - If working with small datasets for example for images < 10K , self supervised might not work good enough for it and ImageNet pretraining might be more beneficial



## Further Reading

For a deeper dive into these concepts, especially semi-supervised techniques, consider reading Lilian Weng's excellent three-part blog series on the subject:  

[Part 1: Semi-Supervised Learning](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/)

[Part 2: Active Learning](https://lilianweng.github.io/posts/2022-02-20-active-learning/)

[Part 3: Data Augmentations](https://lilianweng.github.io/posts/2022-04-15-data-gen/)


## References: 

<a id="1">[1]</a>
Assran, M., Chidlovskii, B., Misra, I., Bojanowski, P., Bordes, A., Rabbat, M., LeCun, Y., Ballas, N. (2023).
"Semi-supervised or Unsupervised? Revisiting Semi-supervised Learning for Vision Models."
arXiv:2307.08919

<a id="2">[2]</a>
Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A., Raffel, C. (2019).
"MixMatch: A Holistic Approach to Semi-Supervised Learning."
arXiv:1905.02249

<a id="3">[3]</a>
Kim, J., Hou, W., Lim, H., Kim, K., Yoo, D. (2021).
"SelfMatch: Combining Contrastive Self-Supervision and Semi-Supervision for Object Detection."
arXiv:2101.06480

<a id="4">[4]</a>
Li, J., Xiong, C., Hoi, S. (2020).
"CoMatch: Semi-supervised Learning with Contrastive Graph Regularization."
arXiv:2011.11183








