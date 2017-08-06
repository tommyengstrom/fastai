import keras
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
import keras.models as M
import keras.optimizers as O
import keras.layers as L
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def main():

    train_gen, val_gen = data_gen(use_sample=True)
    model = build_model()
    # import pudb;pudb.set_trace()
    model.compile(optimizer=O.rmsprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=5,
        validation_data=val_gen,
        validation_steps=50
        )
    import pudb; pudb.set_trace()

def data_gen(use_sample=False, batch_size=8, target_size=(224, 224)):
    base_path = 'data/sample/' if use_sample else 'data/full'
    train_generator = ImageDataGenerator(rescale=1/255)
    test_generator = ImageDataGenerator(rescale=1/255)

    train_gen = train_generator.flow_from_directory(
        base_path + 'train/',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
        )
    val_gen = test_generator.flow_from_directory(
        base_path + 'validation/',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
       )
    return train_gen, val_gen



def build_model():
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    m = L.Flatten()(vgg.output)
    m = L.Dense(100, activation='relu')(m)
    m = L.Dense(1, activation='softmax')(m)
    return M.Model(vgg.input, m)

# initial_model = VGG16(weights="imagenet", include_top=False)
# last = model.output

# x = Flatten()(last)
# x = Dense(1024, activation='relu')(x)
# preds = Dense(200, activation='softmax')(x)

# model = Model(initial_model.input, preds)
#     return vgg

if __name__ == '__main__':
    main()
