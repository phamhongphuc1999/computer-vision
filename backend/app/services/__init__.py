from typing import Sequence

import cv2
import dlib
from sanic.request import Request
from sanic.response import HTTPResponse
from termcolor import colored


def get_img(image_path: str = None, image=None):
    real_image = None
    if image is not None:
        real_image = image
    elif image_path is not None:
        real_image = cv2.imread(image_path)
    if real_image is None:
        raise Exception("image was not found")
    return real_image


def convert_xywh_to_recognition(face_location: Sequence[int]):
    x, y, w, h = face_location
    return y, x + w, y + h, x


def convert_xywh_to_rectangle(face_location: Sequence[int]):
    x, y, w, h = face_location
    return (x, y), (x + w, y + h)


def convert_xywh_to_d_rectangle(face_location: Sequence[int]):
    x, y, w, h = face_location
    drect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    return drect


def rotate_image(image, angle: int):
    if angle == 0:
        return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result


def draw_image(image, rectangle, text: str = None):
    top_left_coordinate, bottom_right_coordinate = rectangle
    cv2.rectangle(image, top_left_coordinate, bottom_right_coordinate, (0, 255, 0), 2)
    if text:
        cv2.putText(
            image,
            text,
            top_left_coordinate,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )


def draw_xywh_image(image, face_location, text: str = None):
    rectangle = convert_xywh_to_rectangle(face_location)
    draw_image(image, rectangle, text)


async def after_request(request: Request, response: HTTPResponse) -> HTTPResponse:
    try:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "deny"
    finally:
        return response


def log(message: str, keyword: str = "WARN"):
    if keyword == "WARN":
        print(colored("[WARN]", "yellow"), message)
    elif keyword == "ERROR":
        print(colored("[ERROR] " + message, "red"))
    elif keyword == "INFO":
        print(colored("[INFO]", "blue"), message)
    else:
        print(colored("[{}]".format(keyword), "cyan"), message)
