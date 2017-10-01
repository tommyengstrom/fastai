import keras.layers as L
import keras.models as M
import keras.optimizers as O
from keras.datasets import imdb

def main():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                          num_words=None,
                                                          skip_top=0,
                                                          maxlen=None,
                                                          seed=113,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3)

if __name__ == "__main__":
    main()







