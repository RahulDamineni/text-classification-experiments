from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.layers import Embedding, Input, Flatten
from keras.models import Model
from data_3d import load_preprocessed_data


SENT_LEN = 100
NUM_SENT = 5
EMBD_DIM = 100
INP_DIM = 22000


if __name__ == "__main__":

    # This is two level, first sentence should be encoded,
    # then this encoded info should be used to further encode document
    # (made of sentences)

    # sentence encoder
    sentence_input = Input(shape=(100,))
    embeddings = Embedding(input_dim=INP_DIM, output_dim=EMBD_DIM,
                           input_length=SENT_LEN)(sentence_input)
    sentence_encoded = Bidirectional(LSTM(64))(embeddings)
    sentence_encoder_model = Model(inputs=sentence_input,
                                   outputs=sentence_encoded)

    # Document encoder
    document_input = Input(shape=(NUM_SENT, SENT_LEN))
    sentences_encoded = TimeDistributed(sentence_encoder_model)(document_input)
    document_encoded = Bidirectional(LSTM(64))(sentences_encoded)

    fully_connected = Dense(32, activation="relu")(document_encoded)
    outputs_ = Dense(1, activation="sigmoid")(fully_connected)

    hierarchial_model = Model(inputs=document_input, outputs=outputs_)
    hierarchial_model.summary()
    hierarchial_model.compile(loss="binary_crossentropy", optimizer="adam",
                              metrics=["accuracy"])

    reviews, labels = load_preprocessed_data()
    hierarchial_model.fit(x=reviews, y=labels, epochs=3, validation_split=0.2)
