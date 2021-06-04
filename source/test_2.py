
import numpy as np
import tensorflow as tf
import cv2

# Loading my model
model = tf.keras.models.model_from_json(open("/home/rants/PycharmProjects/kbs-project/json_files/fer.json", "r").read())

# loading the weights
model.load_weights('/home/rants/PycharmProjects/kbs-project/models/fer.h5')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/rants/PycharmProjects/kbs-project/cascades/data/haarcascade_frontalface_default.xml')

# rectangle details
colour = (255, 0, 255)
font_colour = (255, 0, 0)
stroke = 2

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces_detected:
        end_coordinate_x = x + w
        end_coordinate_y = y + h
        cv2.rectangle(frame, (x, y), (end_coordinate_x, end_coordinate_y), colour, stroke)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        img_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # finding my max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, font_colour)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    # cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


