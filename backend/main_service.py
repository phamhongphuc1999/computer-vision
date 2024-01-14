import cv2

from app.services import rotate_image
from app.services.face_detection import FaceDetection
from app.services.face_extraction import FaceExtraction
from app.services.face_recognition import FaceRecognition

if __name__ == "__main__":
    image = cv2.imread("test/face1.jpeg")

    # face_location, rotate = FaceDetection.rotated_face_locations(image=image)
    # FaceDetection.draw(image, face_location)

    result = FaceDetection.rotated_landmark(image=image)
    FaceDetection.draw_landmark(image, result, 'normal')

    # faces, names = FaceRecognition.recognition('encoding', image=image)
    # FaceRecognition.draw(image, faces, names)

    # known_encodings, known_names = FaceExtraction.extract_by_folder('images')
    # FaceExtraction.save('encoding', known_encodings, known_names)

    cv2.imshow("Output", image)
    cv2.waitKey(0)
