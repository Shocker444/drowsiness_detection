import tensorflow as tf
import cv2

classifier = tf.keras.models.load_model('drowsiness_model')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    for (x, y, w, h) in eyes:
        eye = gray[y:y + h, x:x + w]
        eye = tf.expand_dims(eye, axis=-1)
        eye = tf.image.resize(eye, (96, 96))
        eye = tf.reshape(eye, (96, 96, 1))
        prediction = classifier(eye[tf.newaxis])
        if prediction[0] > 0.5:
            cv2.putText(frame, 'Awake', (18, 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), thickness=2)

        else:
            cv2.putText(frame, 'Drowsy', (18, 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), thickness=2)

        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(225, 0, 0), thickness=2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()