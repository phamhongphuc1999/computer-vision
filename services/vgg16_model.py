import json
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.preprocessing.image import DirectoryIterator, ImageDataGenerator

BATCH_SIZE = 32


def get_vgg16_model(num_classes: int):
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
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=200, activation="relu"))
    model.add(Dense(units=num_classes, activation="softmax"))

    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt)
    return model


def get_dataset(train_path: str, test_path: str):
    trdata = ImageDataGenerator(
        rescale=1.0 / 224, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )
    train_data = trdata.flow_from_directory(
        directory=train_path, target_size=(224, 224)
    )
    tsdata = ImageDataGenerator(rescale=1.0 / 224)
    test_data = tsdata.flow_from_directory(directory=test_path, target_size=(224, 224))
    return train_data, test_data


def save_metadata(train_data: DirectoryIterator, metadata_path: str):
    num_classes = train_data.num_classes
    class_indices = train_data.class_indices
    names = class_indices.keys()
    labels = {}
    for name in names:
        encoded_label = class_indices[name]
        labels[encoded_label] = name
    file = open(metadata_path, "w", encoding="utf-8")
    json.dump(
        {"num_classes": num_classes, "class_indices": class_indices, "labels": labels},
        file,
    )
    file.close()


def load_metadata(metadata_path: str):
    file = open(metadata_path)
    data = json.load(file)
    file.close()
    return data["num_classes"], data["class_indices"], data["labels"]


def execute_model(
    model: Sequential,
    train_data: DirectoryIterator,
    val_data: DirectoryIterator,
    epochs: int,
):
    total_train = train_data.n
    total_val = val_data.n

    steps_per_epoch = np.ceil(total_train / BATCH_SIZE)
    validation_steps = np.ceil(total_val / BATCH_SIZE)

    checkpoint = ModelCheckpoint(
        "vgg16_1.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )
    early = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=20, verbose=1, mode="auto"
    )
    hist = model.fit_generator(
        steps_per_epoch=steps_per_epoch,
        generator=train_data,
        validation_data=val_data,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[checkpoint, early],
    )
    return hist
