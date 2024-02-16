import math
import cv2


def analytic_img(_path: str, faces: list, display_width: int):
    image = cv2.imread(_path)
    face_locations = []
    height, width, channels = image.shape
    new_height = math.floor(display_width * height / width)

    width_ratio = display_width / width
    height_ratio = new_height / height

    for i, face in enumerate(faces):
        x, y, w, h = face["box"]
        face_locations.append(
            {
                "x": x * width_ratio,
                "y": y * height_ratio,
                "w": w * width_ratio,
                "h": h * height_ratio,
                "id": i
            }
        )
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        left_eye = face["keypoints"]["left_eye"]
        right_eye = face["keypoints"]["right_eye"]
        nose = face["keypoints"]["nose"]
        mouth_left = face["keypoints"]["mouth_left"]
        mouth_right = face["keypoints"]["mouth_right"]
        image = cv2.circle(image, left_eye, radius=0, color=(0, 0, 255), thickness=8)
        image = cv2.circle(image, right_eye, radius=0, color=(0, 0, 255), thickness=8)
        image = cv2.circle(image, nose, radius=0, color=(0, 0, 255), thickness=8)
        image = cv2.circle(image, mouth_left, radius=0, color=(0, 0, 255), thickness=8)
        image = cv2.circle(image, mouth_right, radius=0, color=(0, 0, 255), thickness=8)
    image = cv2.resize(image, (display_width, new_height))
    return image, face_locations, new_height
