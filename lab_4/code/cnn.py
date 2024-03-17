import numpy as np
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from utils import load_mnist, draw_plots

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 28
IMG_WIDTH = 28
EPOCHS = 20
BATCH_SIZE = 32


def fully_connected_network():
    model = keras.Sequential()
    model.add(Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def generate_augmented_data(x, num_of_augments):
    data_generator = ImageDataGenerator(width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        fill_mode='nearest')

    output_x = []
    x = x.reshape((1, 28, 28, 1))
    break_iter = 0

    for x_batch in data_generator.flow(x, batch_size=1):
        output_x.append(x_batch)
        break_iter += 1
        if break_iter >= num_of_augments:
            break
    return output_x


def preprocess_data(x_data, y_data, use_augmentation=False, num_of_augments=1):
    x_data = np.array(x_data, dtype='float32').reshape((x_data.shape[0], IMG_HEIGHT, IMG_WIDTH, 1))
    y_data = np.array(y_data)
    x_out = []
    y_out = []
    skip_img = 0
    for x, y in zip(x_data, y_data):
        x /= 255

        if use_augmentation and skip_img % 3 == 0:
            augmented_x = generate_augmented_data(x, num_of_augments)
            for a_x in augmented_x:
                x_out.append(a_x.reshape((IMG_HEIGHT, IMG_WIDTH, 1)))
                y_out.append(y)

        x_out.append(x)
        y_out.append(y)
        skip_img += 1

    return np.array(x_out), to_categorical(y_out)


def get_normalized_data(x_data, y_data):
    x_data = np.array(x_data, dtype='float32')
    x_data = x_data.reshape((x_data.shape[0], 28, 28, 1))
    x_data = x_data / 255

    y_data = to_categorical(y_data)
    return x_data, y_data


def eval_model(model):
    x_train, y_train = load_mnist('data', kind='train')
    x_test, y_test = load_mnist('data', kind='t10k')
    x_train, y_train = preprocess_data(x_train, y_train, use_augmentation=True)
    x_test, y_test = get_normalized_data(x_test, y_test)
    train_hist = model.fit(x_train,
                           y_train,
                           validation_data=(x_train, y_train),
                           batch_size=BATCH_SIZE,
                           epochs=EPOCHS,
                           validation_split=0.2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    draw_plots(train_hist)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)


def process_conv_model():
    model = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # MaxPooling2D((2, 2), strides=2),
        Conv2D(64, (3, 3), activation='relu'),
        # MaxPooling2D((2, 2), strides=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def process_conv_model_dropout():
    model = keras.Sequential([
        Conv2D(32, (3, 3), padding='same',
               activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.3),
        Conv2D(64, (3, 3), padding='same',
               activation='relu'),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def process_conv_model_dropout_batch_norm():
    model = keras.Sequential([
        Conv2D(32, (3, 3), padding='same',
               activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.3),
        Conv2D(64, (3, 3), padding='same',
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # execute only if run as a script
    # eval_model(process_conv_model())
    eval_model(process_conv_model_dropout())
