# text-classification-experiments #
Used 25k reviews from standard IMDB dataset on various models to predict if a given review is positive OR negative.

### What all models have been tried? ###

* Simple 1D convolution model: Passes whole review as a sequence to match a binary output.
* Simple RNN model: Passes whole review as a sequence to learn a review embedding which predicts a binary output.
* Attention RNN model: This time, the model learns which part of review embedding contributes how much to predict output.
* Hierarchial RNN model: Breaks the review into sentences and learns sentence embedding first and then uses them to learn review embedding. 
  Also tries to predict a binary output.
* Hierarchial Attention RNN model: Understands Reviews as list of sentences (sentence embeddings) and learns what sentences weigh how much to
  the final score of predicting whether if a review is positive or negative.

