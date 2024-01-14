import os

import cv2

img = cv2.imread("test/face1.jpeg")

# convert to grayscale of each frames
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(face_path)

eye_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
eye_cascade = cv2.CascadeClassifier(eye_path)

# detects faces in the input image
faces = face_cascade.detectMultiScale(gray, 1.3, 4)
print("Number of detected faces:", len(faces))

# loop over the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_gray = gray[y: y + h, x: x + w]
    roi_color = img[y: y + h, x: x + w]

    # detects eyes of within the detected face area (roi)
    eyes = eye_cascade.detectMultiScale(roi_gray)
    print(eyes)

    # draw a rectangle around eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

# display the image with detected eyes
cv2.imshow("Eyes Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
