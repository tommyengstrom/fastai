import keras.layers as L
import keras.models as M
import keras.optimizers as O
from keras.datasets import imdb
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding

seq_len = 200
maxfeatures = 10000

def load_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                          num_words=maxfeatures -1,
                                                          skip_top=0,
                                                          maxlen=seq_len,
                                                          seed=113,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3)
    pad = sequence.pad_sequences
    return ((pad(x_train, maxlen=seq_len),y_train),
            (pad(x_test,maxlen=seq_len), y_test))

def build_model():
    return Sequential([Embedding(maxfeatures, 25,input_length=seq_len),
                       L.Flatten(),
                      L.Dense(100, activation=K.relu),
                      L.Dense(1,activation=K.sigmoid),
                      ])

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(x_train[0])
    print(x_test)
    print(len(x_train), len(x_test))
    print(len(x_train[0]), len(x_test[0]))
    model.summary()
    model.fit(x_train, y_train,
          batch_size=30,
          epochs=12,
          validation_data=(x_test, y_test))

if __name__ == "__main__":
    main()







