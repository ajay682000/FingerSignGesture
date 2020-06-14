from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

# Loading the trained model
model = load_model('finger_sign.h5')
img_dim = 128
class_labels = ['five', 'four', 'none', 'one', 'three', 'two']

# Starting to capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()
#     Drawing the rectangle
    cv2.rectangle(frame, (100,100), (500,500), (255,255,255), 2)
    roi = frame[100:500, 100:500]
    img = cv2.resize(roi, (img_dim,img_dim))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
    pred = np.argmax(model.predict(img))
    color = (0,0,255)
# Putting the text on the screen
    cv2.putText(frame, class_labels[pred], (50,50), font, 1.0, color, 2)
#     Show the frame
    cv2.imshow('Video', frame)
# Pressing 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()