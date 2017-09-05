import os
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D, MaxPooling1D, Dense
from keras.layers import Embedding, Input, Flatten
from keras.models import Model

SEQ_LEN = 100
EMB_DIM = 100
MAX_FET = 22000


def load_and_preprocess_data():
    with open("/Users/dsp/Documents/AllProjects/Personal/LearningKeras/old_data/testData.p", "rb") as data_file:
        reviews, lables = pickle.load(data_file)

    tokenizer = Tokenizer(num_words=MAX_FET)
    tokenizer.fit_on_texts(reviews)

    reviews_sequences = tokenizer.texts_to_sequences(reviews)
    reviews_sequences = pad_sequences(reviews_sequences, maxlen=SEQ_LEN)

    with open("preprocessedTestData.p", "wb+") as data_out:
        pickle.dump([reviews_sequences, np.array(lables)], data_out)

    return [reviews_sequences, np.array(lables)]


def load_preprocessed_data():
    if os.path.isfile("preprocessedTestData.p") is True:
        with open("preprocessedTestData.p", "rb") as input_file:
            return pickle.load(input_file)
    else:
        return load_and_preprocess_data()


if __name__ == "__main__":
    load_preprocessed_data()
