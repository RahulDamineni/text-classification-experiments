from data import load_preprocessed_data
from keras.layers import LSTM, Bidirectional, Dense
from keras.layers import Embedding, Input
from keras.models import Model

INPUT_DIM = 22000
SENT_LEN = 1000
EMBD_DIM = 100


if __name__ == "__main__":
    inputs_, outputs_ = load_preprocessed_data()

    inputs = Input(shape=(SENT_LEN,), dtype="int32")
    embeddings = Embedding(input_dim=INPUT_DIM, output_dim=EMBD_DIM,
                           input_length=SENT_LEN)(inputs)
    lstm1 = LSTM(units=512, activation="softmax")(embeddings)
    outputs = Dense(1, activation="softmax")(lstm1)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss="binary_crossentropy", optimizer="rmsprop",
                  metrics=["accuracy"])
    model.fit(x=inputs_, y=outputs_, validation_split=0.15, batch_size=512,
              epochs=3)
