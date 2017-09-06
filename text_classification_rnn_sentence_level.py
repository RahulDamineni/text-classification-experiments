from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.layers import Embedding, Input, Flatten
from keras.models import Model
from data_3d import load_preprocessed_data


SENT_LEN = 50
NUM_SENT = 20
EMBD_DIM = 100
INP_DIM = 22000


if __name__ == "__main__":

    # This is two level, first sentence should be encoded,
    # then this encoded info should be used to further encode document
    # (made of sentences)

    # sentence encoder
    sentence_input = Input(shape=(SENT_LEN,))
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

    reviews, labels = load_preprocessed_data(sent_len=SENT_LEN,
                                             num_sent=NUM_SENT)
    hierarchial_model.fit(x=reviews, y=labels, epochs=3, validation_split=0.2)

# Analysis on dataset revealed average number of sentences per review is 14
# ADAM with only 64 LSTM cells and input of (5, 100)
# Epoch 2/3
# 20000/20000 [==============================] - 375s - loss: 0.2540
#  - acc: 0.8993 - val_loss: 0.4173 - val_acc: 0.8180
# Epoch 3/3
# 20000/20000 [==============================] - 363s - loss: 0.1402
#  - acc: 0.9482 - val_loss: 0.5134 - val_acc: 0.8156
