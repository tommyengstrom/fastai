from keras.applications.vgg16 import VGG16
import keras.models as M
import keras.optimizers as O
import keras.layers as L
from keras.preprocessing.image import ImageDataGenerator

def main():
    img_size = (224, 224)

    train_gen, val_gen = data_gen(use_sample=False, img_size=img_size)
    model = build_simple_model(img_size=img_size)
    model.compile(optimizer=O.adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit_generator(
        train_gen,
        steps_per_epoch=200,
        epochs=10,
        validation_data=val_gen,
        validation_steps=5
        )
    import pudb; pudb.set_trace()
    model.save_weights('model_weights.hdf5')

def data_gen(img_size, use_sample=False, batch_size=32):
    base_path = 'data/sample/' if use_sample else 'data/full/'
    image_generator = ImageDataGenerator() #rescale=1/255)

    train_gen = image_generator.flow_from_directory(
        base_path + 'train/',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
        )
    val_gen = image_generator.flow_from_directory(
        base_path + 'validation/',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
       )
    return train_gen, val_gen


def build_simple_model(img_size):
    return M.Sequential(
        [ L.Conv2D(16, (3, 3), input_shape=img_size + (3,))
        , L.Activation('relu')
        , L.BatchNormalization(axis=1)
        , L.MaxPooling2D(pool_size=(2, 2))
        , L.Conv2D(32, (3, 3))
        , L.Activation('relu')
        , L.BatchNormalization(axis=1)
        , L.MaxPooling2D(pool_size=(2, 2))
        , L.Conv2D(64, (3, 3))
        , L.Activation('relu')
        , L.BatchNormalization(axis=1)
        , L.MaxPooling2D(pool_size=(2, 2))
        , L.Flatten()
        , L.BatchNormalization()
        , L.Dense(256, activation='relu')
        , L.BatchNormalization()
#        , L.Dropout(0.5)
        , L.Dense(2, activation='softmax')
        ])

def build_vgg_model(img_size):
    # vgg = VGG16(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    vgg = VGG16(weights='imagenet', include_top=True, input_shape=img_size + (3,))
    for layer in vgg.layers:
        layer.trainable = False
    # m = L.Flatten()(vgg.output)
    m = vgg.output
    # m = L.Dense(100, activation='relu')(m)
    m = L.Dense(2, activation='softmax')(m)
    return M.Model(vgg.input, m)

if __name__ == '__main__':
    main()
