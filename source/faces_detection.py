import time

import cv2 as cv
import numpy as np
import tensorflow as tf

# Loading my model
model = tf.keras.models.model_from_json(open("json_files/fer.json", "r").read())

# loading the weights
model.load_weights('models/fer.h5')

face_cascade = cv.CascadeClassifier('/home/rants/PycharmProjects/kbs-project/cascades/data/haarcascade_frontalface_default.xml')
# face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)

# rectangle details
colour = (255, 0, 255)
font_colour = (255, 0, 0)
stroke = 2

while True:
    # capturing video frame by frame
    ret, frame = cap.read()
    if not ret:
        continue
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces_detected:
        end_coordinate_x = x + w
        end_coordinate_y = y + h
        # print(x, y, w, h)
        cv.rectangle(frame, (x, y), (end_coordinate_x, end_coordinate_y), colour, stroke)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)
        img_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # finding my max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv.putText(frame, predicted_emotion, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, font_colour)
        # time.sleep(5)

        resized_img = cv.resize(frame, (1000, 700))
        cv.imshow('frame', frame)

        # img_item = 'my-image.png'
        # cv.imwrite(img_item, roi_gray)

    # displaying the resulting image
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# releasing the capture
cap.release()
cv.destroyAllWindows()
