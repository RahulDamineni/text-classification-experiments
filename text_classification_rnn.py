from data import load_preprocessed_data
from keras.layers import LSTM, Bidirectional, Dense, SimpleRNN
from keras.layers import Embedding, Input, Dropout
from keras.models import Model

INPUT_DIM = 22000
SENT_LEN = 100
EMBD_DIM = 128


if __name__ == "__main__":
    inputs_, outputs_ = load_preprocessed_data()

    inputs = Input(shape=(SENT_LEN,), dtype="int32")
    embeddings = Embedding(input_dim=INPUT_DIM, output_dim=EMBD_DIM,
                           input_length=SENT_LEN)(inputs)
    lstm1 = Bidirectional(LSTM(units=64))(embeddings)
    dropout = Dropout(0.35)(lstm1)
    outputs = Dense(1, activation="sigmoid")(dropout)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.fit(x=inputs_, y=outputs_, validation_split=0.20, batch_size=64,
              epochs=3)


# ADAM optimizer
# Epoch 3/3
# 20000/20000 [==============================] - 183s - loss: 0.3373 -
# acc: 0.8891 - val_loss: 0.4146 - val_acc: 0.8274

# RMSPROP optimizer
# Epoch 3/3
# 21250/21250 [==============================] - 287s - loss: 0.3246 -
# acc: 0.8913 - val_loss: 0.3591 - val_acc: 0.8579

# ADAM optimizer with dropout 0.35 (too much to generalize?)
# Epoch 3/3
# 20000/20000 [==============================] - 131s - loss: 0.1092 -
# acc: 0.9625 - val_loss: 0.4861 - val_acc: 0.8422
