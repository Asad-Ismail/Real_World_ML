# Real World ML: From Theory to Production

<p align="center">
  <img src="images/logo.png" alt="Real World ML" width="20%" height="20%">
</p>

<p align="center">
  <strong>Learn machine learning by building it from scratch, then applying it to solve real-world problems</strong>
</p>

## Overview

This repository provides comprehensive machine learning implementations built from first principles, combined with production-ready examples for real-world deployment.

**Learn by Implementation**: Every algorithm is built from scratch using minimal dependencies, helping you understand the mathematics and intuition behind ML/DL techniques.

**Production-Ready Examples**: Bridge the gap between academic understanding and real-world deployment with complete end-to-end pipelines.

**Comprehensive Coverage**: From classical ML to cutting-edge deep learning, covering supervised/unsupervised learning, NLP, computer vision, and reinforcement learning.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Asad-Ismail/Real_World_ML.git
cd Real_World_ML

# Install dependencies
pip install -r requirements.txt

# Try a quick example
python learn/Supervised/LogisticRegression/logisticregression.py
```

## Repository Structure

### [`/learn/`](./learn/) - Algorithm Implementations

**Supervised Learning** [`/learn/Supervised/`](./learn/Supervised/)
- [Decision Trees](./learn/Supervised/DecisionTree/) - From scratch tree building with various splitting criteria
- [Ensemble Methods](./learn/Supervised/ensemble/) - Gradient boosting, random forests with custom implementations
- [Linear Models](./learn/Supervised/LinearRegression/) - Linear/logistic regression with regularization
- [Support Vector Machines](./learn/Supervised/svm/) - SVM implementation with different kernels
- [Naive Bayes](./learn/Supervised/NaiveBayes/) - Probabilistic classification
- [k-NN](./learn/Supervised/knn/) - Instance-based learning

**Deep Learning**
- [CNNs](./learn/CNNs/) - Convolutional neural networks from scratch
- [LLMs from Scratch](./learn/LLM_from_scratch/) - Transformer architecture, attention mechanisms, BPE tokenization
- [Generative Models](./learn/Generative_Models/) - GANs, VAEs, diffusion models, NeRF implementations

**Unsupervised Learning** [`/learn/Unsupervised/`](./learn/Unsupervised/)
- [PCA](./learn/Unsupervised/pca/) - Principal component analysis
- [t-SNE](./learn/Unsupervised/t-SNE/) - Dimensionality reduction and visualization
- [K-Means](./learn/Unsupervised/kmeans/) - Clustering algorithms
- [Autoencoders](./learn/Unsupervised/autoencoder/) - Neural network-based dimensionality reduction

**Natural Language Processing** [`/learn/NLP/`](./learn/NLP/)
- [BERT Implementation](./learn/NLP/BERT/) - Transformer-based language model
- [Word2Vec](./learn/NLP/word2vec/) - Skip-gram and CBOW implementations
- [Tokenizers](./learn/NLP/tokenizers/) - Text preprocessing and tokenization

**Reinforcement Learning** [`/learn/Reinforcement_Learning/`](./learn/Reinforcement_Learning/)
- [Q-Learning & SARSA](./learn/Reinforcement_Learning/TD/) - Temporal difference methods
- [Policy Gradient](./learn/Reinforcement_Learning/policygrad/) - REINFORCE, Actor-Critic, A2C, SAC
- [Custom Environments](./learn/Reinforcement_Learning/envs/) - Grid world and other RL environments

**Specialized Topics**
- [Graph Neural Networks](./learn/GNN/) - DGL-based fraud detection pipeline
- [Active Learning](./learn/active_learning/) - Smart data labeling strategies
- [Explainable AI](./learn/explainable_interpretable_AI/) - GradCAM, saliency maps, interpretability tools
- [Time Series](./learn/TimeSeries/) - Forecasting and temporal data analysis
- [Probability Theory](./learn/probability/) - Bayesian methods, Kalman filtering, sensor fusion

### [`/Use_Cases/`](./Use_Cases/) - Production Examples

**[Real-Time Data Processing](./Use_Cases/RealTimeDataProcessing/)**
- Complete Kafka + Spark streaming pipeline
- ML model inference on streaming data
- Scalable architecture for production deployment

**[AWS SageMaker End-to-End](./Use_Cases/Sagmeaker_Fraud_Detection_End_To_End/)**
- Complete fraud detection pipeline
- Model training, deployment, and monitoring
- Lambda functions for real-time inference

**[Spark Image Processing](./Use_Cases/SparkImageProcessing/)**
- Distributed image processing with PySpark
- Scalable computer vision workflows

**[Learning with Less Data](./Use_Cases/learning_with_less/)**
- Comprehensive guide to data-efficient learning
- Transfer learning, semi-supervised, and active learning strategies
- Performance comparisons and best practices

## Learning Paths

### Beginner Path: Start with Fundamentals
1. [Linear Regression](./learn/Supervised/LinearRegression/) → [Logistic Regression](./learn/Supervised/LogisticRegression/)
2. [Decision Trees](./learn/Supervised/DecisionTree/) → [Random Forest](./learn/Supervised/ensemble/)
3. [K-Means](./learn/Unsupervised/kmeans/) → [PCA](./learn/Unsupervised/pca/)

### Intermediate Path: Deep Learning & NLP
1. [CNNs](./learn/CNNs/) → [Generative Models](./learn/Generative_Models/)
2. [Word2Vec](./learn/NLP/word2vec/) → [BERT](./learn/NLP/BERT/)
3. [LLM Components](./learn/LLM_from_scratch/) → [Transformer Architecture](./learn/LLM_from_scratch/)

### Advanced Path: Production & Specialized Topics
1. [Real-Time Processing](./Use_Cases/RealTimeDataProcessing/) → [AWS SageMaker Pipeline](./Use_Cases/Sagmeaker_Fraud_Detection_End_To_End/)
2. [Graph Neural Networks](./learn/GNN/) → [Active Learning](./learn/active_learning/)
3. [Reinforcement Learning](./learn/Reinforcement_Learning/) → [Explainable AI](./learn/explainable_interpretable_AI/)

## Technical Requirements

**Core Dependencies:**
- Python 3.7+
- NumPy, Matplotlib, Scikit-learn
- PyTorch (for deep learning examples)
- Additional dependencies listed in `requirements.txt`

**For Production Examples:**
- Apache Kafka (Real-time processing)
- Apache Spark/PySpark (Distributed processing)
- AWS CLI (SageMaker examples)
- Docker (Containerized deployments)

## Key Features

- **Educational Focus**: Every implementation includes detailed comments explaining the mathematics
- **From Scratch Implementation**: Minimal external dependencies - understand every line of code
- **Comprehensive Testing**: Most implementations include test cases and validation examples
- **Production Ready**: Complete pipelines from data ingestion to model deployment
- **Real-World Applications**: Tackle fraud detection, image processing, NLP, and time series forecasting

## Contributing

We welcome contributions including:
- Bug fixes and performance improvements
- Enhanced documentation and examples
- New algorithm implementations
- Additional production use cases

Please feel free to open issues and pull requests.

## Additional Resources

- **Detailed Explanations**: Check the [learning_with_less](./Use_Cases/learning_with_less/) directory for comprehensive guides
- **Research References**: Most implementations include links to original papers and theoretical foundations
- **Best Practices**: Production examples demonstrate industry-standard practices and deployment patterns

## Contact

For questions, suggestions, or discussions about machine learning concepts, please open an issue in this repository.
