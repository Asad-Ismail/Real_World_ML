## WORD2VEC

Skip-Gram and Continuous Bag of Words (CBOW) are two different architectures used for training Word2Vec, an unsupervised learning algorithm that represents words as high-dimensional vectors. Both methods aim to learn word embeddings by predicting words from their context, but they have slightly different approaches:

Skip-Gram: Given a target word, this method predicts the surrounding context words within a specified window size. In other words, it learns to represent the target word by predicting the context in which it appears. Skip-Gram performs well with large datasets and can capture rare words effectively due to its focus on individual context words.

CBOW (Continuous Bag of Words): CBOW predicts the target word given its surrounding context words. It takes the average of the context word embeddings and uses it to predict the target word. This method is faster to train than Skip-Gram and works well for small datasets and frequent words. However, it may not capture rare words as effectively as Skip-Gram.

In summary, the main differences between Skip-Gram and CBOW are:

Prediction objective: Skip-Gram predicts context words given a target word, while CBOW predicts the target word given context words.
Training speed: CBOW is generally faster to train, as it averages the context word embeddings, while Skip-Gram processes each context word separately.
Performance: Skip-Gram performs better with large datasets and rare words, while CBOW is more suited for small datasets and frequent words.