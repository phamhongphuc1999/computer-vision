import cv2
import face_recognition
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.src.applications.vgg16 import decode_predictions

from services.face_recognition_model import get_encodings


def vgg16_model():
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


def face_recognition_model():
    known_encodings, known_names = get_encodings('resources/abcccc')

    img = cv2.imread("resources/test/elon_musk/elon_musk1.jpg")
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(rgb_image, known_face_locations=[(0, 224, 224, 0)])
    matches = face_recognition.compare_faces(known_encodings, encoding[0])
    name = 'unknown'
    face_distance = face_recognition.face_distance(known_encodings, encoding[0])
    best_match_index = np.argmin(face_distance)
    print("best_match_index", best_match_index)
    if matches[best_match_index]:
        name = known_names[best_match_index]
    print("name", name)


if __name__ == "__main__":
    pass
