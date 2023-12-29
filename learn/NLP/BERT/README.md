## BERT (Bidirectional Encoder Representations from Transformers)

BERT is trained using two main objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).

Masked Language Modeling (MLM): In this task, some percentage of the input data is masked at random, and the model is trained to predict the masked words based on the context provided by the non-masked words. For example, in the sentence "The cat sat on the ___", the word "mat" might be masked and the model would be trained to predict "mat" based on the context provided by the rest of the sentence. This allows the model to learn a bidirectional representation of the sentence, as it must consider the context from both the left and the right of the masked word in order to make its prediction.

Next Sentence Prediction (NSP): In this task, the model is trained to predict whether a sentence B is the actual next sentence that follows sentence A in the original document. During training, 50% of the inputs are a pair in which sentence B is the actual next sentence that follows sentence A, and 50% of the inputs are a pair in which sentence B is a random sentence from the corpus. For example, given two sentences "I went to the store. I bought some milk.", the model would be trained to predict that the second sentence does indeed follow the first. But if the sentences were "I went to the store. The Eiffel Tower is in Paris.", the model would be trained to predict that the second sentence does not logically follow the first.

These two training objectives allow BERT to understand the context of words in a sentence (via MLM) and the relationships between sentences (via NSP), which makes it a powerful model for a wide range of natural language processing tasks.




'''
def BERT_objective_function(model, tokens, is_next_label):
    """
    model: The BERT model.
    tokens: The input tokens.
    is_next_label: Binary label indicating whether the second sentence in the pair is the actual next sentence.
    """

    # Get the predictions for the masked tokens and the next sentence prediction
    masked_token_predictions, next_sentence_prediction = model(tokens)

    # Compute the MLM loss
    MLM_loss = -torch.mean(torch.log(masked_token_predictions))

    # Compute the NSP loss
    NSP_loss = -torch.log(next_sentence_prediction[is_next_label])

    # The total loss is the sum of the MLM loss and the NSP loss
    total_loss = MLM_loss + NSP_loss

    return total_loss

'''

## BERT Vs BART

BERT (Bidirectional Encoder Representations from Transformers) and BART (Bidirectional and Auto-Regressive Transformers) are both transformer-based models developed for natural language processing tasks, but they have different architectures and are used for different types of tasks.

BERT (Bidirectional Encoder Representations from Transformers): BERT is a transformer-based model that uses a bidirectional training approach. This means that it learns contextual relations between words in a text by looking at the words that come before and after a given word. BERT is pre-trained on a large corpus of text and then fine-tuned for specific tasks, such as question answering, named entity recognition, or sentiment analysis. BERT's architecture consists only of the transformer's encoder part.

BART (Bidirectional and Auto-Regressive Transformers): BART is also a transformer-based model, but it is structured differently from BERT. BART is a denoising autoencoder. During pre-training, it corrupts the input by randomly masking out some tokens (words), and then it tries to reconstruct the original input. BART is used for both understanding and generation tasks, such as text summarization, translation, and conversation generation. Unlike BERT, BART uses both the encoder and the decoder parts of the transformer.

In summary, the main differences between BERT and BART are:

Architecture: BERT uses only the encoder part of the transformer, while BART uses both the encoder and decoder parts.
Training Objective: BERT is trained to predict masked words in a sentence, while BART is trained to reconstruct the original sentence from a corrupted version.
Use Cases: BERT is typically used for understanding tasks, while BART can be used for both understanding and generation tasks.