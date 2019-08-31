import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import os
from os import listdir
from keras.preprocessing.image import img_to_array


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation,Flatten,Dropout,Dense


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



from keras.models import load_model


classifier=load_model('emotion_little_vgg_3.h5')



class_labels=val_generator.class_indices
class_labels={v:k for k,v in class_labels.items()}
classes=list(class_labels.values())

print(class_labels)



face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def face_detector(img):
    gray=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    

    allfaces=[]
    rects=[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x,w,y,h))
    return rects, allfaces, img





def func(rects,faces,image):
    i=0

    for face in faces:
        roi=face.astype('float')/255.0
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)

        #prediction

        preds=classifier.predict(roi)[0]
        label=class_labels[preds.argmax()]

        label_position=(rects[i][0]+int((rects[i][1]/2)),abs(rects[i][2]-10))
        i=+1
        cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)


    cv2.imshow("Emotion Detector", image)
    cv2.waitKey(0)

img1=cv2.imread('happy.jpg')
img2=cv2.imread('angry.jpg')


rects,faces,image=face_detector(img1)
func(rects,faces,image)

rects,faces,image=face_detector(img2)

func(rects,faces,image)




cv2.destroyAllWindows()
