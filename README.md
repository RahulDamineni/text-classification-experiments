# text-classification-experiments #
Used 25k reviews from standard IMDB dataset on various models to predict if a given review is positive OR negative.

## What all models have been tried? ##

* **Simple 1D convolution model (`1Dconv_simple_noatt`)**: Passes whole review as a sequence to match a binary output.
* **Simple RNN model (`RNN_simple_noatt`)**: Passes whole review as a sequence to learn a review embedding which predicts a binary output.
* **Attention RNN model (`RNN_simple_att`)**: This time, the model learns which part of review embedding contributes how much to predict output.
* **Hierarchial RNN model (`RNN_hier_noatt`)**: Breaks the review into sentences and learns sentence embedding first and then uses them to learn review embedding. 
  Also tries to predict a binary output.
* **Hierarchial Attention RNN model (`RNN_hier_att`)**: Understands Reviews as list of sentences (sentence embeddings) and learns what sentences weigh how much to
  the final score of predicting whether if a review is positive or negative.


## Results

|Model|Dropout|Input Sequence Dims.|Best Validation Accuracy|Time To Train|Optimizer|Attention|Hierarchy|
|:---:|:-----:|:------------------:|:----------------------:|:-----------:|:-------:|:-------:|:-------:|
|**1Dconv_simple_noatt**|No|(21250, 1000)|88.19%|181s|`rmsprop`|`None`|`None`|
|**1Dconv_simple_noatt**|Yes|(21250, 1000)|85.39%|183s|`rmsprop`|`None`|`None`|
|**RNN_simple_noatt**|No|(21250,100)|85.79%|287s|`rmsprop`|`None`|`None`|
|**RNN_simple_noatt**|No|(20000,100)|82.74%|183s|`ADAM`|`None`|`None`|
|**RNN_simple_noatt**|Yes|(20000,100)|84.22%|131s|`ADAM`|`None`|`None`|
|**RNN_simple_noatt**|No|(20000,500)|87.18%|415s|`ADAM`|`None`|`None`|
|**RNN_simple_noatt**|Yes|(20000,500)|87.34%|608s|`ADAM`|`None`|`None`|
|**RNN_simple_noatt**|Yes|(20000,1000)|86.56%|1234s|`ADAM`|`None`|`None`|



