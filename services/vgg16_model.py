import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.preprocessing.image import DirectoryIterator, ImageDataGenerator

from utils import count_file_and_folder

BATCH_SIZE = 32


def get_vgg16_model(number_classes: int):
    model = Sequential()
    model.add(
        Conv2D(
            input_shape=(224, 224, 3),
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(
        Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(
        Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
    )
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=number_classes, activation="softmax"))

    opt = Adam(lr=0.001)
    model.compile(
        optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=["accuracy"]
    )
    model.summary()
    return model


def get_dataset(train_path: str, test_path: str):
    trdata = ImageDataGenerator()
    train_data = trdata.flow_from_directory(
        directory=train_path, target_size=(224, 224)
    )
    tsdata = ImageDataGenerator()
    test_data = tsdata.flow_from_directory(directory=test_path, target_size=(224, 224))
    return train_data, test_data


def execute_model(
    model: Sequential,
    train_data: DirectoryIterator,
    test_data: DirectoryIterator,
    epochs: int,
    period: int,
):
    _, total_train = count_file_and_folder("./resources/mini_faces")
    _, total_test = count_file_and_folder("./resources/mini_test")
    train_steps_per_epoch = np.ceil((total_train * 0.8 / BATCH_SIZE) - 1)
    val_steps_per_epoch = np.ceil((total_test * 0.2 / BATCH_SIZE) - 1)
    print("train_steps_per_epoch", train_steps_per_epoch)
    print("val_steps_per_epoch", val_steps_per_epoch)

    checkpoint = ModelCheckpoint(
        "vgg16_1.h5",
        monitor="val_acc",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq=period,
    )
    early = EarlyStopping(
        monitor="val_acc", min_delta=0, patience=20, verbose=1, mode="auto"
    )
    hist = model.fit_generator(
        steps_per_epoch=14,
        generator=train_data,
        validation_data=test_data,
        validation_steps=1,
        epochs=epochs,
        callbacks=[checkpoint, early],
    )
    return hist
