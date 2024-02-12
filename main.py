from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.src.applications.vgg16 import decode_predictions


def abc():
    img = image.load_img(
        "resources/test/zoe_saldana/zoe_saldana1.jpg",
        target_size=(224, 224),
    )
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    saved_model = load_model("resources/vgg16_1.h5")
    output = saved_model.predict(img)
    result = decode_predictions(output)
    print("result", result)
    _max = output[0][0]
    _index = 0
    for i, item in enumerate(output[0]):
        if item > _max:
            _max = item
            _index = i
    print("_max", output, _max, _index)


if __name__ == "__main__":
    abc()
