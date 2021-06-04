
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/rants/PycharmProjects/kbs-project/cascades/data/haarcascade_frontalface_default.xml')

# rectangle details
colour = (255, 0, 255)
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
        print(x, y, w, h)
        cv2.rectangle(frame, (x, y), (end_coordinate_x, end_coordinate_y), colour, stroke)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

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


