# Active Learning Strategies for Deep Learning

We show three active learning strategies designed for deep learning tasks, specifically for obejct detection and segmentation. Active learning is a semi-supervised machine learning approach that strategically selects the most informative unlabeled samples for labeling. This process aims to reduce labeling costs while maximizing model performance. The strategies implemented here are:


1. **Object score for Object Detection**
2. **Monte Carlo Dropout for Image Segmentation**
3. **Entropy-based Selection for Image Segmentation**



## 1. Object Score for Object Detection

This method utilizes inherent score predicted for each object by the model to assign scores to unlabelled dataset. It is easiest method to implemnt but if the DNN is not calibrated well the scores might be too close together as Deep learning models are highly overconfident. 



## 1. Monte Carlo Dropout for Image Segmentation

This method utilizes Monte Carlo (MC) dropout to estimate model uncertainty during inference, even with dropout layers active. By performing multiple forward passes and applying dropout each time, we obtain varied predictions that reflect the model's uncertainty about its outputs. The variance across these predictions serves as a measure of uncertainty, helping us select the most informative samples for labeling.

## 2. Entropy-based Selection for Image Segmentation

This strategy calculates the entropy of the model's predictions to gauge uncertainty. Higher entropy values indicate greater uncertainty, making those samples prime candidates for labeling. This approach leverages the softmax outputs of a segmentation model to compute entropy directly.
