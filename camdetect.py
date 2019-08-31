import cv2
import numpy as np
from time import sleep
from keras.preprocessing.image import img_to_array


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation,Flatten,Dropout,Dense


from keras.models import load_model
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

num_classes=7
img_row,img_col=48,48

batch_size=16

train_data_dir="/home/canok/Desktop/TransferLearning/EmotionDetector/fer2013/train"

val_data_dir="/home/canok/Desktop/TransferLearning/EmotionDetector/fer2013/validation"



tran_datagen=ImageDataGenerator(

    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode="nearest"
)


val_datagen=ImageDataGenerator(rescale=1./255)

train_generator=tran_datagen.flow_from_directory(
    train_data_dir,
    color_mode="grayscale",
    target_size=(img_row,img_col),batch_size=batch_size
)

val_generator=val_datagen.flow_from_directory(
    val_data_dir,
    color_mode="grayscale",
    target_size=(img_row,img_col),batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)



classifier=load_model('emotion_little_vgg_3.h5')


class_labels=val_generator.class_indices
class_labels={v:k for k,v in class_labels.items()}
classes=list(class_labels.values())


face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((48,48), np.uint8), img
    return (x,w,y,h), roi_gray, img

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]  
        label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
        cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 3)
    else:
        cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,250), 3)
        
    cv2.imshow('All', image)
    key = cv2.waitKey(1)
    if key == 27:#ESC
        cv2.destroyAllWindows()
        break
        
